import tensorflow as tf
import tensorflow.contrib.layers as layers

from pysc2.lib import actions
import numpy as np
import threading
import time
import utils as U

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Softmax
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Lambda;
import tensorflow.keras.backend as K

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
        self.train_queue = [[], [], [], [],[],[]]


        self.counter_lock = threading.Lock()
        self.training_counter = 0

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        K.set_session(self.session)
        K.manual_variable_initialization(True)
        self.build_net('/gpu:0')
        self.build_model('/gpu:0')
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()
        self.summary_writer.add_graph(self.default_graph)

    def stop(self):
        self.stop_signal = True

    def getPredictFeedDict(self, observation, hState, cState):
        input = np.expand_dims(np.array(self.preprocess(observation), dtype=np.float32), axis=0)
        hState = np.expand_dims(np.array(hState), axis=0)
        cState = np.expand_dims(np.array(cState), axis=0)
        return {
            self.input: input,
            self.hStateInput: hState,
            self.cStateInput: cState,
        }

    def predict(self, feed):
        with self.session.as_default():
            policy, value, hs, cs = self.model.predict([
                feed[self.input],
                feed[self.hStateInput],
                feed[self.cStateInput],
                feed[self.xB],
                feed[self.yB],
                feed[self.divB]
            ])
            return policy, value, hs, cs


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
            for i in range(len(self.train_queue)):
                self.train_queue[i] = self.train_queue[i] + batch[i]
            if len(self.train_queue[0]) > (5000 * Brain.MIN_BATCH):
                print("Training queue too large; optimizer likely crashed %d" % len(self.train_queue[0]))
                exit()

    def optimize(self):
        time.sleep(0.001)

        with self.lock_queue:
            if len(self.train_queue[0]) < Brain.MIN_BATCH:	# more thread could have passed without lock
                return 									# we can't yield inside lock

            batch = self.train_queue
            self.train_queue = [[],[],[],[],[],[]]

        batch_train_feed = {
            self.value_target:np.squeeze(np.array(batch[0], dtype=np.float32)),
            self.input:np.array(batch[1], dtype=np.float32),
            self.action_selected:np.array(batch[2], dtype=np.float32),
            self.advantage:np.array(batch[3], dtype=np.float32),
            self.hStateInput:np.array(batch[4], dtype=np.float32),
            self.cStateInput:np.array(batch[5], dtype=np.float32),
        }
        self.train(batch_train_feed)

    def getPolicyLoss(self, action_probability, advantage):
        return -tf.log(action_probability + 1e-10) * advantage

    def getValueLoss(self, difference):
        return tf.square(difference)

    def getEntropy(self, policy):
        return tf.reduce_sum(policy * tf.log(policy + 1e-10), axis=1)


    def build_model(self, dev):
        # Set targets and masks
        with tf.variable_scope('a3c'):
            self.action_selected = tf.placeholder(
                tf.float32, [None, self.output_shape[0]], name='action_selected')
            self.value_target = tf.placeholder(
                tf.float32, [None], name='value_target')
            self.entropy_rate = tf.placeholder(
                tf.float32, None, name='entropy_rate')
            self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
            self.advantage = tf.placeholder(tf.float32, [None], name='advantage')
            if len(self.input_shape) > 1:
                self.input = tf.placeholder(tf.float32, [None, 80,80,1], name='input')
            else:
                self.input = tf.placeholder(tf.float32, [None, self.input_shape[0]], name='input')
            self.hStateInput = tf.placeholder(
                tf.float32, [None, self.NUM_LSTM], name='h_state_input')
            self.cStateInput = tf.placeholder(
                tf.float32, [None, self.NUM_LSTM], name='c_state_input')

            self.policy, self.value, _, _ = self.model([self.input,self.hStateInput,self.cStateInput])

            # This will get the probability of choosing a valid action. Given that we force it to choose from
            # the set of valid actions. The probability of an action is the probability the policy chooses
            # divided by the probability of a valid action
            action_probability = tf.reduce_sum(
                self.action_selected * self.policy, axis=1)

            policy_loss = self.getPolicyLoss(action_probability, self.advantage)
            value_loss = self.getValueLoss(self.value_target - self.value)
            entropy = self.getEntropy(self.policy)

            loss = tf.reduce_mean(policy_loss + value_loss * .5 + entropy *.01)

            global_step = tf.Variable(0)
            learning_rate_decayed = tf.train.exponential_decay(self.learning_rate,
            global_step,10000, .95, staircase=True)

            # Build the optimizer
            opt = tf.train.AdamOptimizer(learning_rate_decayed)
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

    def expand_dims(self, x):
        return K.expand_dims(x, 1)

    def Squeeze(self, x):
        return K.squeeze(x,axis=-1)

    def build_net(self, dev):
        with tf.variable_scope('a3c'):
            if len(self.input_shape) > 1:
                input = Input(shape = (80,80,1),name='Input')
                conv1 = Conv2D(32, kernel_size=8, strides=(4,4),padding='same',activation='relu',name='conv1')(input)
                conv2 = Conv2D(64, kernel_size=4, strides=(2,2),padding='same',activation='relu',name='conv2')(conv1)
                conv3 = Conv2D(64, kernel_size=3, strides=(1,1),padding='same',activation='relu',name='conv3')(conv2)
                flatConv3 = Flatten(name='flatConv3')(conv3)
                fc1 = Dense(512,activation='relu')(flatConv3)
            else:
                input = Input(shape =(self.input_shape[0],),name='Input')
                fc1 = Dense(512, activation='relu')(input)

            LSTMIN = Lambda(self.expand_dims, name='expandFc1')(fc1)

            self.NUM_LSTM = 100
            hStateInput = Input(
                shape=(self.NUM_LSTM,),
                name='hStateInput'
            )

            cStateInput = Input(
                shape=(self.NUM_LSTM,),
                name='cStateInput'
            )

            lstm, hStates, cStates = LSTM(self.NUM_LSTM, return_state=True)(LSTMIN,initial_state=[hStateInput, cStateInput])

            fc2 = Dense(256, activation='relu',name='fc2')(lstm)
            vFc3 = Dense(1, activation='linear',name='vFc3')(fc2)

            value = Lambda(self.Squeeze, name='value')(vFc3)
            policy = Dense(self.output_shape[0], activation='softmax',name='policy')(fc2)

            self.model = Model(
                inputs=[input, hStateInput, cStateInput],
                outputs=[policy, value, hStates, cStates]
            )
            self.model._make_predict_function()
