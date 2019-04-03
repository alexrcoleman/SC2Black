import tensorflow as tf
import tensorflow.contrib.layers as layers

from pysc2.lib import actions
import numpy as np
import threading
import time
import utils as U

class Brain:
    MIN_BATCH = 25
    def __init__(self, flags, summary_writer, input_shape, output_shape):
        self.lr = flags.learning_rate
        self.er = flags.entropy_rate
        self.flags = flags
        self.summary_writer = summary_writer
        self.N_STEP_RETURN = 20
        self.GAMMA = .99
        self.LAMBDA = 1
        self.stop_signal = False
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.lock_queue = threading.Lock()
        self.train_queue = [[], [], [], []]


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
        input = np.expand_dims(np.array(self.preprocess(observation), dtype=np.float32), axis=0)
        return {
            self.input: input,}

    def predict(self, feed):
        policy, value = self.session.run(
        [self.action, self.value],
        feed_dict=feed)
        return policy, value


    def preprocess(self, data):
        if len(data.shape) == 1:
            return data
        stacks = []
        for obs in data:
            cropped = obs[34:194, :, :]
            reduced = cropped[0:-1:2, 0:-1:2]
            grayscale = np.sum(reduced, axis=2)
            bw = np.zeros(grayscale.shape)
            bw[grayscale != 233] = 1
            expand = np.expand_dims(bw, axis=2)
            stacks.append(expand)
        return np.concatenate(stacks, axis=2)

    def train(self, feed):
        feed[self.learning_rate] = self.lr
        feed[self.entropy_rate] = self.er

        _, summary = self.session.run([self.train_op, self.summary_op], feed_dict=feed)
        if self.flags.use_tensorboard:
            with self.counter_lock:
                self.training_counter = self.training_counter + 1
                self.summary_writer.add_summary(summary, self.training_counter)
                self.summary_writer.flush()

    def add_train(self, batch):
        with self.lock_queue:
            self.train_queue[0] = self.train_queue[0] + batch[0]
            self.train_queue[1] = self.train_queue[1] + batch[1]
            self.train_queue[2] = self.train_queue[2] + batch[2]
            self.train_queue[3] = self.train_queue[3] + batch[3]
            if len(self.train_queue[0]) > (5000 * Brain.MIN_BATCH):
                print("Training queue too large; optimizer likely crashed %d" % len(self.train_queue[0]))
                exit()

    def optimize(self):
        time.sleep(0.001)

        with self.lock_queue:
            if len(self.train_queue[0]) < Brain.MIN_BATCH:	# more thread could have passed without lock
                return 									# we can't yield inside lock

            batch = self.train_queue
            self.train_queue = [[],[],[],[]]
        # print("BATCH", batch[0], batch[1], batch[2], "END")
        if len(batch) > 5000*Brain.MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(batch))
        batch_train_feed = {
            self.value_target:np.squeeze(np.array(batch[0], dtype=np.float32)),
            self.input:np.array(batch[1], dtype=np.float32),
            self.action_selected:np.array(batch[2], dtype=np.float32),
            self.advantage:np.array(batch[3], dtype=np.float32),
        }
        self.train(batch_train_feed)

    def getPolicyLoss(self, action_probability, advantage):
        return -tf.log(action_probability + 1e-10) * advantage

    def getValueLoss(self, difference):
        return tf.square(difference)

    def getEntropy(self, policy):
        return tf.reduce_sum(policy * tf.log(policy + 1e-10), axis=1)


    def build_model(self, dev):
        with tf.variable_scope('a3c') and tf.device(dev):
            # Set targets and masks
            self.action_selected = tf.placeholder(
                tf.float32, [None, self.output_shape[0]], name='action_selected')
            self.value_target = tf.placeholder(
                tf.float32, [None], name='value_target')
            self.entropy_rate = tf.placeholder(
                tf.float32, None, name='entropy_rate')
            self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
            self.advantage = tf.placeholder(tf.float32, [None], name='advantage')

            # This will get the probability of choosing a valid action. Given that we force it to choose from
            # the set of valid actions. The probability of an action is the probability the policy chooses
            # divided by the probability of a valid action
            action_probability = tf.reduce_sum(
                self.action_selected * self.action, axis=1)

            policy_loss = self.getPolicyLoss(action_probability, self.advantage)
            value_loss = self.getValueLoss(self.value_target - self.value)
            entropy = self.getEntropy(self.action)

            loss = tf.reduce_mean(policy_loss + value_loss * .5 + entropy *.01)

            # Build the optimizer
            opt = tf.train.AdamOptimizer(self.learning_rate)
            grads, vars = zip(*opt.compute_gradients(loss))
            grads, glob_norm = tf.clip_by_global_norm(grads, 40.0)
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
                    'advantage', tf.reduce_mean(self.advantage)))
                summary.append(tf.summary.scalar(
                    'loss', tf.reduce_mean(loss)))
                self.summary_op = tf.summary.merge(summary)
            else:
                self.summary_op = []


    def build_net(self, dev):
        with tf.variable_scope('a3c') and tf.device(dev):
            if len(self.input_shape) > 1:
                self.input = tf.placeholder(
                    tf.float32, [None, 80,80,4], name='input')
                conv = layers.conv2d(self.input,
                                       num_outputs=32,
                                       kernel_size=8,
                                       stride=4,
                                       scope='sconv1',
                                       activation_fn=tf.nn.relu)
                conv = layers.conv2d(conv,
                                       num_outputs=64,
                                       kernel_size=4,
                                       stride=2,
                                       scope='sconv2',
                                       activation_fn=tf.nn.relu)
                conv = layers.conv2d(conv,
                                       num_outputs=64,
                                       kernel_size=3,
                                       stride=1,
                                       scope='sconv3',
                                       activation_fn=None)
                fc = layers.fully_connected(layers.flatten(conv),
                                             num_outputs=512,
                                             activation_fn=tf.nn.relu,
                                             scope='info_fc')

            else:
                self.input = tf.placeholder(
                    tf.float32, [None, self.input_shape[0]], name='input')


                fc = layers.fully_connected(self.input,
                                             num_outputs=16,
                                             activation_fn=tf.nn.relu,
                                             scope='info_fc')

            self.action = layers.fully_connected(fc,
                                                        num_outputs=self.output_shape[0],
                                                        activation_fn=tf.nn.softmax,
                                                        scope='action')

            self.value = tf.reshape(layers.fully_connected(fc,
                                                      num_outputs=1,
                                                      activation_fn=None,
                                                      scope='value'), [-1])
