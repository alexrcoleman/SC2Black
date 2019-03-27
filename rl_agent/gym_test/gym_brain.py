import tensorflow as tf
import tensorflow.contrib.layers as layers

from pysc2.lib import actions
import numpy as np
import threading
import time
import utils as U
class Brain:
    MIN_BATCH = 32
    def __init__(self, flags, summary_writer, input_shape, output_shape):
        self.lr = flags.learning_rate
        self.er = flags.entropy_rate
        self.flags = flags
        self.summary_writer = summary_writer
        self.N_STEP_RETURN = 8
        self.GAMMA = self.flags.discount
        self.GAMMA_N = self.GAMMA**self.N_STEP_RETURN
        self.stop_signal = False
        self.input_shape = input_shape
        self.output_shape = output_shape

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

    def getPredictFeedDict(self, observation):
        input = np.expand_dims(np.array(observation, dtype=np.float32), axis=0)
        return {
            self.input: input,}

    def predict(self, feed):
        policy, value = self.session.run(
        [self.action, self.value],
        feed_dict=feed)
        return policy, value

    def getTrainFeedDict(self, observation, reward, action_id):
        input = np.expand_dims(np.array(observation, dtype=np.float32), axis=0)
        value_target = np.zeros([1], dtype=np.float32)
        value_target[0] = reward
        action_selected = np.zeros(
            [1, self.output_shape[0]], dtype=np.float32)
        action_selected[0, action_id] = 1
        return {
            self.input:input,
            self.value_target: value_target,
            self.action_selected: action_selected,
        }


    def train(self, feed):
        feed[self.learning_rate] = self.lr
        feed[self.entropy_rate] = self.er

        _, summary = self.session.run([self.train_op, self.summary_op], feed_dict=feed)
        if self.flags.use_tensorboard:
            with self.counter_lock:
                self.training_counter = self.training_counter + 1
                self.summary_writer.add_summary(summary, self.training_counter)
                self.summary_writer.flush()

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
            if len(self.train_queue[0]) < Brain.MIN_BATCH:	# more thread could have passed without lock
                return 									# we can't yield inside lock

            tfs, pfs, t_mask = self.train_queue
            self.train_queue = [ [], [], [] ]

        if len(tfs) > 5*Brain.MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(tfs))
        batch_predict_feed = U.make_batch(pfs)
        _, v = self.predict(batch_predict_feed)

        batch_train_feed = U.make_batch(tfs)

        r = batch_train_feed[self.value_target]
        r = r + self.GAMMA_N * v * np.array(t_mask)
        batch_train_feed[self.value_target] = r

        self.train(batch_train_feed)

    def getPolicyLoss(self, action_probability, advantage):
        return -tf.log(action_probability + 1e-10) * tf.stop_gradient(advantage)

    def getValueLoss(self, advantage):
        return tf.square(advantage)

    def getEntropy(self, policy):
        return tf.reduce_sum(policy * tf.log(policy + 1e-10))


    def build_model(self, dev):
        with tf.variable_scope('a3c') and tf.device(dev):
            # Set targets and masks
            self.action_selected = tf.placeholder(
                tf.float32, [None, self.output_shape[0]], name='action_selected')
            self.value_target = tf.placeholder(
                tf.float32, [None], name='value_target')
            self.entropy_rate = tf.placeholder(
                tf.float32, None, name='entropy_rate')

            # This will get the probability of choosing a valid action. Given that we force it to choose from
            # the set of valid actions. The probability of an action is the probability the policy chooses
            # divided by the probability of a valid action
            action_probability = tf.reduce_sum(
                self.action_selected * self.action)

            # The advantage function, which will represent how much better this action was than what was expected from this state
            advantage = self.value_target - self.value

            policy_loss = self.getPolicyLoss(action_probability, advantage)
            value_loss = self.getValueLoss(advantage)
            entropy = self.getEntropy(self.action)

            loss = tf.reduce_mean(policy_loss + value_loss * .5 + entropy * self.entropy_rate)

            # Build the optimizer
            self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
            opt = tf.train.AdamOptimizer(self.learning_rate)
            grads, vars = zip(*opt.compute_gradients(loss))
            grads, glob_norm = tf.clip_by_global_norm(grads, 280.0)
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


    def build_net(self, dev):
        with tf.variable_scope('a3c') and tf.device(dev):
            self.input = tf.placeholder(
                tf.float32, [None, self.input_shape[0]], name='input')


            fc = layers.fully_connected(self.input,
                                         num_outputs=16,
                                         activation_fn=tf.nn.relu,
                                         scope='info_fc')

            self.action = layers.fully_connected(fc,
                                                        num_outputs=self.output_shape[0],
                                                        activation_fn=tf.nn.softmax,
                                                        scope='non_spatial_action')

            self.value = tf.reshape(layers.fully_connected(fc,
                                                      num_outputs=1,
                                                      activation_fn=None,
                                                      scope='value'), [-1])
