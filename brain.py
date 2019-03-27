import tensorflow as tf
import tensorflow.contrib.layers as layers

from pysc2.lib import actions
import utils as U
import numpy as np
import threading
import time

class Brain:
    MIN_BATCH = 25
    MIN_EPS = .01
    MAX_EPS = .2
    def __init__(self, flags, summary_writer):
        self.lr = flags.learning_rate
        self.er = flags.entropy_rate
        self.eps = Brain.MAX_EPS
        self.flags = flags
        self.summary_writer = summary_writer
        self.N_STEP_RETURN = 10
        self.GAMMA = self.flags.discount
        self.GAMMA_N = self.GAMMA**self.N_STEP_RETURN
        self.ssize = 64
        self.isize = len(U.useful_actions)
        self.custom_input_size = 1 + len(U.useful_actions)
        self.stop_signal = False

        self.lock_queue = threading.Lock()
        self.train_queue = [[], [], []]


        self.counter_lock = threading.Lock()
        self.training_counter = 0

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        self.build_net('/gpu:0')
        self.build_model('/gpu:0')

        init_op = tf.global_variables_initializer()
        self.session.run(init_op)

    def stop(self):
        self.stop_signal = True

    def getPredictFeedDict(self, obs):
        screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
        screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
        info = np.zeros([1, len(U.useful_actions)], dtype=np.float32)
        info[0, U.compressActions(obs.observation['available_actions'])] = 1
        custom_inputs = np.expand_dims(
            np.array(obs.observation.custom_inputs, dtype=np.float32), axis=0)

        return {
            self.screen: screen,
            self.info: info,
            self.custom_inputs: custom_inputs, }

    def getTrainFeedDict(self, obs, action, attributed_act_id):
        screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
        screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
        info = np.zeros([1, len(U.useful_actions)], dtype=np.float32)
        info[0, U.compressActions(obs.observation['available_actions'])] = 1
        value_target = np.zeros([1], dtype=np.float32)
        value_target[0] = obs.reward
        valid_spatial_action = np.zeros([1], dtype=np.float32)
        spatial_action_selected = np.zeros([1, self.ssize**2], dtype=np.float32)
        valid_non_spatial_action = np.zeros([1, len(U.useful_actions)], dtype=np.float32)
        non_spatial_action_selected = np.zeros(
            [1, len(U.useful_actions)], dtype=np.float32)
        custom_inputs = np.expand_dims(
            np.array(obs.observation.custom_inputs, dtype=np.float32), axis=0)

        act_id = action.function
        net_act_id = attributed_act_id
        act_args = action.arguments

        player_relative = obs.observation.feature_screen.player_relative
        valid_actions = obs.observation["available_actions"]
        valid_actions = U.compressActions(valid_actions)
        valid_non_spatial_action[0, valid_actions] = 1
        non_spatial_action_selected[0, net_act_id] = 1

        args = actions.FUNCTIONS[act_id].args
        for arg, act_arg in zip(args, act_args):
            if arg.name in ('screen', 'minimap', 'screen2') and (not self.flags.force_focus_fire or (act_id != 12 and act_id != 2)):
                ind = act_arg[1] * self.ssize + act_arg[0]
                valid_spatial_action[0] = 1
                spatial_action_selected[0, ind] = 1
        return {
            self.screen: screen,
            self.info: info,
            self.custom_inputs: custom_inputs,
            self.value_target: value_target,
            self.valid_spatial_action: valid_spatial_action,
            self.spatial_action_selected: spatial_action_selected,
            self.valid_non_spatial_action: valid_non_spatial_action,
            self.non_spatial_action_selected: non_spatial_action_selected,
        }

    def predict(self, feed):
        nspatial, spatial, v = self.session.run(
            [self.non_spatial_action, self.spatial_action, self.value],
            feed_dict=feed)
        return nspatial, spatial, v

    def train(self, feed):
        feed[self.learning_rate] = self.lr * (1 - 0.9 * self.training_counter / self.flags.max_train_steps)
        feed[self.entropy_rate] = self.er * (1 - 0.5 * self.training_counter / self.flags.max_train_steps)

        _, summary = self.session.run([self.train_op, self.summary_op], feed_dict=feed)
        with self.counter_lock:
            self.eps = max(self.eps * .999, Brain.MIN_EPS)
            local_counter = self.training_counter
            self.training_counter = self.training_counter + 1
        if self.flags.use_tensorboard:
            self.summary_writer.add_summary(summary, local_counter)
            self.summary_writer.flush()
            if local_counter % self.flags.snapshot_step == 1 or local_counter >= self.flags.max_train_steps:
                self.save_model('./snapshot/'+self.flags.map, local_counter)
                print("Snapshot of model saved at", local_counter)

            if local_counter >= self.flags.max_train_steps:
                print("Reached step %d, training complete." % local_counter)
                self.stop()

    def add_train(self, train_feed, predict_feed, t_mask):
        with self.lock_queue:
            self.train_queue[0].append(train_feed)
            self.train_queue[1].append(predict_feed)
            self.train_queue[2].append(t_mask)
            if len(self.train_queue[0]) > 5000:
                print("Training queue too large; optimizer likely crashed")
                exit()

    def optimize(self):
        time.sleep(0.001)
        if len(self.train_queue[0]) < Brain.MIN_BATCH:
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < Brain.MIN_BATCH:
                return

            tfs, pfs, t_mask = self.train_queue
            self.train_queue = [ [], [], [] ]

        if len(tfs) > 5*Brain.MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(tfs))
        batch_predict_feed = U.make_batch(pfs)
        _, _, v = self.predict(batch_predict_feed)

        batch_train_feed = U.make_batch(tfs)

        r = batch_train_feed[self.value_target]
        r = r + self.GAMMA_N * v * np.array(t_mask)
        batch_train_feed[self.value_target] = r

        self.train(batch_train_feed)

    def getPolicyLoss(self, action_probability, advantage):
        return -tf.log(action_probability + 1e-10) * tf.stop_gradient(advantage)

    def getValueLoss(self, advantage):
        return tf.square(advantage)

    def getEntropy(self, policy, spatial_policy, valid_spatial):
        return tf.reduce_sum(policy * tf.log(policy + 1e-10), axis=1) + tf.reduce_sum(spatial_policy * tf.log(spatial_policy + 1e-10), axis=1)

    def getMinRoachHealthLoss(self, roach_target, roach_prediction):
        return tf.reduce_sum(tf.square(roach_target - roach_prediction), axis=1)

    def build_model(self, dev):
        with tf.variable_scope('a3c') and tf.device(dev):
            # Set targets and masks
            self.valid_spatial_action = tf.placeholder(
                tf.float32, [None], name='valid_spatial_action')
            self.spatial_action_selected = tf.placeholder(
                tf.float32, [None, self.ssize**2], name='spatial_action_selected')
            self.valid_non_spatial_action = tf.placeholder(
                tf.float32, [None, len(U.useful_actions)], name='valid_non_spatial_action')
            self.non_spatial_action_selected = tf.placeholder(
                tf.float32, [None, len(U.useful_actions)], name='non_spatial_action_selected')
            self.value_target = tf.placeholder(
                tf.float32, [None], name='value_target')
            self.entropy_rate = tf.placeholder(
                tf.float32, None, name='entropy_rate')

            # This will get the probability of choosing a valid action. Given that we force it to choose from
            # the set of valid actions. The probability of an action is the probability the policy chooses
            # divided by the probability of a valid action
            valid_non_spatial_action_prob = tf.reduce_sum(
                self.valid_non_spatial_action * self.non_spatial_action+1e-10, axis=1)
            non_spatial_action_prob = tf.reduce_sum(
                self.non_spatial_action_selected * self.non_spatial_action+1e-10, axis=1) / valid_non_spatial_action_prob

            # Here we compute the probability of the spatial action. If the action selected was non spactial,
            # the probability will be one.
            # TODO: Make this use vectorized things (using a constant "valid_spatial_action" seems fishy to me, but maybe it's fine)
            spatial_action_prob = (self.valid_spatial_action * tf.reduce_sum(
                self.spatial_action * self.spatial_action_selected, axis=1)) + (1.0 - self.valid_spatial_action)+1e-10

            # The probability of the action will be the the product of the non spatial and the spatial prob
            action_probability = non_spatial_action_prob * spatial_action_prob

            # The advantage function, which will represent how much better this action was than what was expected from this state
            advantage = self.value_target - self.value

            policy_loss = self.getPolicyLoss(action_probability, advantage)
            value_loss = self.getValueLoss(advantage)
            entropy = self.getEntropy(
                self.non_spatial_action, self.spatial_action, self.valid_spatial_action)

            loss = tf.reduce_mean(policy_loss + value_loss * .5 + entropy * self.entropy_rate)

            # Build the optimizer
            self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
            opt = tf.train.AdamOptimizer(self.learning_rate)
            grads, vars = zip(*opt.compute_gradients(loss))
            grads, glob_norm = tf.clip_by_global_norm(grads, 50.0)
            self.train_op = opt.apply_gradients(zip(grads, vars))
            if self.flags.use_tensorboard:
                summary = []
                summary.append(tf.summary.scalar(
                    'policy_loss', tf.reduce_mean(policy_loss)))
                summary.append(tf.summary.scalar(
                    'glob_norm', glob_norm))
                summary.append(tf.summary.scalar(
                    'value_loss', tf.reduce_mean(value_loss)))
                summary.append(tf.summary.scalar(
                    'entropy_loss', tf.reduce_mean(entropy)))
                summary.append(tf.summary.scalar(
                    'advantage', tf.reduce_mean(advantage)))
                summary.append(tf.summary.scalar(
                    'loss', tf.reduce_mean(loss)))
                self.summary_op = tf.summary.merge(summary)
            else:
                self.summary_op = []
            self.saver = tf.train.Saver(max_to_keep=100)

            # debug graph:
            #self.summary_writer.add_graph(self.session.graph)

    def build_net(self, dev):
        with tf.variable_scope('a3c') and tf.device(dev):
            self.screen = tf.placeholder(
                tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
            self.info = tf.placeholder(
                tf.float32, [None, self.isize], name='info')
            self.custom_inputs = tf.placeholder(
                tf.float32, [None, self.custom_input_size], name='custom_input')
            sconv1 = layers.conv2d(tf.transpose(self.screen, [0, 2, 3, 1]),
                                   num_outputs=16,
                                   kernel_size=5,
                                   stride=1,
                                   scope='sconv1',
                                   activation_fn=tf.nn.relu)
            sconv2 = layers.conv2d(sconv1,
                                   num_outputs=32,
                                   kernel_size=3,
                                   stride=1,
                                   scope='sconv2',
                                   activation_fn=tf.nn.relu)

            info_fc = layers.fully_connected(tf.concat([self.info, self.custom_inputs], axis=1),
                                             num_outputs=64,
                                             activation_fn=tf.nn.relu,
                                             scope='info_fc')

            # Compute spatial actions

            self.spatial_action = layers.conv2d(sconv2,
                                           num_outputs=1,
                                           kernel_size=1,
                                           stride=1,
                                           activation_fn=None,
                                           scope='spatial_action')
            self.spatial_action = tf.nn.softmax(layers.flatten(self.spatial_action))
            #self.roach_prediction = layers.conv2d(sconv2,
            #                                 num_outputs=1,
            #                                 kernel_size=1,
            #                                 stride=1,
            #                                 activation_fn=None,
            #                                 scope='roach_prediction')
            #self.roach_prediction = layers.flatten(self.roach_prediction)

            # Compute non spatial actions and value
            feat_fc = tf.concat([layers.flatten(sconv2), info_fc], axis=1)
            feat_fc = layers.fully_connected(feat_fc,
                                             num_outputs=int(256),
                                             activation_fn=tf.nn.relu,
                                             scope='feat_fc')
            self.non_spatial_action = layers.fully_connected(feat_fc,
                                                        num_outputs=self.isize,
                                                        activation_fn=tf.nn.softmax,
                                                        scope='non_spatial_action')
            self.value = tf.reshape(layers.fully_connected(feat_fc,
                                                      num_outputs=1,
                                                      activation_fn=None,
                                                      scope='value'), [-1])

    def save_model(self, path, count):
        self.saver.save(self.session, path + '/model.pkl', count)

    def load_model(self, path):
        print()
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.session, ckpt.model_checkpoint_path)
        self.training_counter = int(ckpt.model_checkpoint_path.split('-')[-1])
