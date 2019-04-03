import tensorflow as tf
import tensorflow.contrib.layers as layers

from pysc2.lib import actions
import utils as U
import numpy as np
import threading
import time

class Brain:
    MIN_BATCH = 25
    def __init__(self, flags, summary_writer):
        self.lr = flags.learning_rate
        self.er = flags.entropy_rate
        self.flags = flags
        self.summary_writer = summary_writer
        self.N_STEP_RETURN = 20
        self.GAMMA = .99
        self.LAMBDA = 1

        self.ssize = 64
        self.isize = len(U.useful_actions)
        self.custom_input_size = 1 + len(U.useful_actions)
        self.stop_signal = False

        self.lock_queue = threading.Lock()
        self.train_queue = [[], [], [], [], []]


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
        screen = U.preprocess_screen(screen)
        info = np.zeros([len(U.useful_actions)], dtype=np.float32)
        info[U.compressActions(obs.observation['available_actions'])] = 1
        valid_spatial_action = 0
        valid_non_spatial_action = np.zeros([len(U.useful_actions)], dtype=np.float32)

        custom_inputs = np.array(obs.observation.custom_inputs, dtype=np.float32)

        act_id = action.function
        net_act_id = attributed_act_id
        act_args = action.arguments

        player_relative = obs.observation.feature_screen.player_relative
        valid_actions = obs.observation["available_actions"]
        valid_actions = U.compressActions(valid_actions)
        valid_non_spatial_action[valid_actions] = 1

        args = actions.FUNCTIONS[act_id].args
        for arg, act_arg in zip(args, act_args):
            if arg.name in ('screen', 'minimap', 'screen2') and (not self.flags.force_focus_fire or (act_id != 12 and act_id != 2)):
                valid_spatial_action = 1

        return {
            self.screen: screen, # yes
            self.info: info, # yes
            self.custom_inputs: custom_inputs, #yes
            self.valid_spatial_action: valid_spatial_action, #yes
            self.valid_non_spatial_action: valid_non_spatial_action, # yes
        }

    def predict(self, feed):
        nspatial, spatial, v = self.session.run(
            [self.non_spatial_action, self.spatial_action, self.value],
            feed_dict=feed)
        return nspatial, spatial, v

    def train(self, feed):
        feed[self.learning_rate] = self.lr
        feed[self.entropy_rate] = self.er

        _, summary = self.session.run([self.train_op, self.summary_op], feed_dict=feed)
        with self.counter_lock:
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

    def add_train(self, batch):
        with self.lock_queue:
            self.train_queue[0] = self.train_queue[0] + batch[0]
            self.train_queue[1] = self.train_queue[1] + batch[1]
            self.train_queue[2] = self.train_queue[2] + batch[2]
            self.train_queue[3] = self.train_queue[3] + batch[3]
            self.train_queue[4] = self.train_queue[4] + batch[4]
            if len(self.train_queue[0]) > (5000 * Brain.MIN_BATCH):
                print("Training queue too large; optimizer likely crashed")
                exit()

    def optimize(self):
        time.sleep(0.001)

        with self.lock_queue:
            if len(self.train_queue[0]) < Brain.MIN_BATCH:
                return

            batch = self.train_queue
            self.train_queue = [[],[],[],[],[]]


        batch_train_feed = {
            self.value_target:np.squeeze(np.array(batch[0], dtype=np.float32)),
            self.screen:np.asarray([x[self.screen] for x in batch[1]],dtype=np.float32),
            self.info:np.asarray([x[self.info] for x in batch[1]],dtype=np.float32),
            self.custom_inputs:np.asarray([x[self.custom_inputs] for x in batch[1]],dtype=np.float32),
            self.valid_spatial_action:np.asarray([x[self.valid_spatial_action] for x in batch[1]],dtype=np.float32),
            self.valid_non_spatial_action:np.asarray([x[self.valid_non_spatial_action] for x in batch[1]],dtype=np.float32),
            self.non_spatial_action_selected:np.array(batch[2], dtype=np.float32),
            self.spatial_action_selected:np.array(batch[3], dtype=np.float32),
            self.advantage:np.array(batch[4], dtype=np.float32),
        }

        self.train(batch_train_feed)

    def getPolicyLoss(self, action_probability, advantage):
        return -tf.log(action_probability + 1e-10) * advantage

    def getValueLoss(self, difference):
        return tf.square(difference)

    def getEntropy(self, policy, spatial_policy, valid_spatial):
        return tf.reduce_sum(policy * tf.log(policy + 1e-10), axis=1) + tf.reduce_sum(spatial_policy * tf.log(spatial_policy + 1e-10), axis=1)

    # def getMinRoachHealthLoss(self, roach_target, roach_prediction):
    #     return tf.reduce_sum(tf.square(roach_target - roach_prediction), axis=1)

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
            self.advantage = tf.placeholder(tf.float32, [None], name='advantage')
            self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')


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


            policy_loss = self.getPolicyLoss(action_probability, self.advantage)
            value_loss = self.getValueLoss(self.value_target - self.value)
            entropy = self.getEntropy(
                self.non_spatial_action, self.spatial_action, self.valid_spatial_action)

            loss = tf.reduce_mean(policy_loss + value_loss * .5 + entropy * .01)
            # Build the optimizer
            global_step = tf.Variable(0)
            learning_rate_decayed = tf.train.exponential_decay(self.learning_rate,
            global_step,10000, .95, staircase=True)

            opt = tf.train.AdamOptimizer(self.learning_rate)
            grads, vars = zip(*opt.compute_gradients(loss))
            grads, glob_norm = tf.clip_by_global_norm(grads, 40.0)
            self.train_op = opt.apply_gradients(zip(grads, vars), global_step=global_step)
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
                    'advantage', tf.reduce_mean(self.advantage)))
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
