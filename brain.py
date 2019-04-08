import tensorflow as tf

from pysc2.lib import actions
import utils as U
import numpy as np
import threading
import time

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
    def __init__(self, flags, summary_writer):
        self.lr = flags.learning_rate
        self.er = flags.entropy_rate
        self.flags = flags
        self.summary_writer = summary_writer
        self.N_STEP_RETURN = 40
        self.GAMMA = .99
        self.LAMBDA = 1
        self.eps = .2
        self.ssize = 32
        self.isize = len(U.useful_actions)
        self.custom_input_size = 1 + len(U.useful_actions)
        self.stop_signal = False

        self.lock_queue = threading.Lock()
        self.train_queue = [[], [], [], [], [], [], [], []]


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
        # self.debug = np.array([])

    def stop(self):
        self.stop_signal = True

    def getPredictFeedDict(self, obs, hState, cState):
        screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
        screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
        info = np.zeros([1, len(U.useful_actions)], dtype=np.float32)
        info[0, U.compressActions(obs.observation['available_actions'])] = 1
        custom_inputs = np.expand_dims(
            np.array(obs.observation.custom_inputs, dtype=np.float32), axis=0)
        hState = np.expand_dims(np.array(hState), axis=0)
        cState = np.expand_dims(np.array(cState), axis=0)
        return {
            self.screen: screen,
            self.info: info,
            self.custom_inputs: custom_inputs,
            self.hStateInput: hState,
            self.cStateInput: cState }

    def getTrainFeedDict(self, obs, action, attributed_act_id):
        screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
        screen = U.preprocess_screen(screen)
        info = np.zeros([len(U.useful_actions)], dtype=np.float32)
        info[U.compressActions(obs.observation['available_actions'])] = 1
        valid_spatial_action = 0
        valid_action = np.zeros([len(U.useful_actions)], dtype=np.float32)

        custom_inputs = np.array(obs.observation.custom_inputs, dtype=np.float32)

        act_id = action.function
        net_act_id = attributed_act_id
        act_args = action.arguments

        player_relative = obs.observation.feature_screen.player_relative
        valid_actions = obs.observation["available_actions"]
        valid_actions = U.compressActions(valid_actions)
        valid_action[valid_actions] = 1

        args = actions.FUNCTIONS[act_id].args
        for arg, act_arg in zip(args, act_args):
            if arg.name in ('screen', 'minimap', 'screen2') and (not self.flags.force_focus_fire or (act_id != 12 and act_id != 2)):
                valid_spatial_action = 1

        return {
            self.screen: screen, # yes
            self.info: info, # yes
            self.custom_inputs: custom_inputs, #yes
            self.valid_spatial_action: valid_spatial_action, #yes
            self.valid_action: valid_action, # yes
        }

    def predict(self, feed):
        with self.session.as_default():
            v, policy, spatialPolicy, hS, cS, _ = self.model.predict([
                feed[self.screen],
                feed[self.info],
                feed[self.custom_inputs],
                feed[self.hStateInput],
                feed[self.cStateInput],
            ])

            return policy, spatialPolicy, v, hS, cS


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
            for i in range(len(self.train_queue)):
                self.train_queue[i] = self.train_queue[i] + batch[i]

            if len(self.train_queue[0]) > (5000 * Brain.MIN_BATCH):
                print("Training queue too large; optimizer likely crashed")
                exit()

    def optimize(self):
        time.sleep(0.001)

        with self.lock_queue:
            if len(self.train_queue[0]) < Brain.MIN_BATCH:
                return

            batch = self.train_queue
            self.train_queue = [[],[],[],[],[],[],[],[]]


        batch_train_feed = {
            self.value_target:np.squeeze(np.array(batch[0], dtype=np.float32)),
            self.screen:np.asarray([x[self.screen] for x in batch[1]],dtype=np.float32),
            self.info:np.asarray([x[self.info] for x in batch[1]],dtype=np.float32),
            self.custom_inputs:np.asarray([x[self.custom_inputs] for x in batch[1]],dtype=np.float32),
            self.valid_spatial_action:np.asarray([x[self.valid_spatial_action] for x in batch[1]],dtype=np.float32),
            self.valid_action:np.asarray([x[self.valid_action] for x in batch[1]],dtype=np.float32),
            self.action_selected:np.array(batch[2], dtype=np.float32),
            self.spatial_action_selected:np.array(batch[3], dtype=np.float32),
            self.advantage:np.array(batch[4], dtype=np.float32),
            self.hStateInput:np.array(batch[5], dtype=np.float32),
            self.cStateInput:np.array(batch[6], dtype=np.float32),
            self.roach_location:np.array(batch[7], dtype=np.float32)
        }

        self.train(batch_train_feed)

    def getPolicyLoss(self, action_probability, advantage):
        return -tf.log(action_probability + 1e-10) * advantage

    def getValueLoss(self, difference):
        return tf.square(difference)

    def getEntropy(self, policy, spatial_policy, valid_spatial):
        return tf.reduce_sum(policy * tf.log(policy + 1e-10), axis=1)# + valid_spatial * tf.reduce_sum(spatial_policy * tf.log(spatial_policy + 1e-10), axis=1)

    def getMinRoachLoss(self, roach_target, roach_prediction):
        return K.categorical_crossentropy(roach_target,roach_prediction)

    def build_model(self, dev):
        with tf.variable_scope('a3c') and tf.device(dev):
            self.valid_spatial_action = tf.placeholder(
                tf.float32, [None], name='valid_spatial_action')
            self.spatial_action_selected = tf.placeholder(
                tf.float32, [None, self.ssize**2], name='spatial_action_selected')
            self.valid_action = tf.placeholder(
                tf.float32, [None, len(U.useful_actions)], name='valid_action')
            self.action_selected = tf.placeholder(
                tf.float32, [None, len(U.useful_actions)], name='action_selected')
            self.value_target = tf.placeholder(
                tf.float32, [None], name='value_target')
            self.entropy_rate = tf.placeholder(
                tf.float32, None, name='entropy_rate')
            self.advantage = tf.placeholder(tf.float32, [None], name='advantage')
            self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
            self.screen = tf.placeholder(
                tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
            self.info = tf.placeholder(
                tf.float32, [None, self.isize], name='info')
            self.custom_inputs = tf.placeholder(
                tf.float32, [None, self.custom_input_size], name='custom_input')
            self.hStateInput = tf.placeholder(
                tf.float32, [None, self.NUM_LSTM], name='h_state_input')
            self.cStateInput = tf.placeholder(
                tf.float32, [None, self.NUM_LSTM], name='c_state_input')
            self.roach_location = tf.placeholder(
                tf.float32, [None, self.ssize ** 2], name='roach_location'
            )

            self.value, self.policy, self.spatial_policy, _, _, self.roachPrediction = self.model([self.screen, self.info, self.custom_inputs, self.hStateInput, self.cStateInput])
            # This will get the probability of choosing a valid action. Given that we force it to choose from
            # the set of valid actions. The probability of an action is the probability the policy chooses
            # divided by the probability of a valid action
            valid_action_prob = tf.reduce_sum(
                self.valid_action * self.policy+1e-10, axis=1)
            action_prob = tf.reduce_sum(
                self.action_selected * self.policy+1e-10, axis=1) / valid_action_prob

            # Here we compute the probability of the spatial action. If the action selected was non spactial,
            # the probability will be one.
            # TODO: Make this use vectorized things (using a constant "valid_spatial_action" seems fishy to me, but maybe it's fine)
            spatial_action_prob = (self.valid_spatial_action * tf.reduce_sum(
                self.spatial_policy * self.spatial_action_selected, axis=1)) + (1.0 - self.valid_spatial_action)+1e-10

            # The probability of the action will be the the product of the non spatial and the spatial prob
            combined_action_probability = action_prob * spatial_action_prob

            # The advantage function, which will represent how much better this action was than what was expected from this state


            policy_loss = self.getPolicyLoss(combined_action_probability, self.advantage)
            value_loss = self.getValueLoss(self.value_target - self.value)
            entropy = self.getEntropy(
                self.policy, self.spatial_policy, self.valid_spatial_action)
            roachLoss = self.getMinRoachLoss(
                self.roach_location, self.roachPrediction
            )


            loss = tf.reduce_mean(policy_loss + value_loss * .5 + entropy * .01 + .5 * roachLoss)
            # Build the optimizer
            global_step = tf.Variable(0)
            learning_rate_decayed = tf.train.exponential_decay(self.learning_rate,
                                                               global_step,10000, .95)

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
                summary.append(tf.summary.scalar(
                    'roachLoss', tf.reduce_mean(roachLoss)))
                self.summary_op = tf.summary.merge(summary)
            else:
                self.summary_op = []
            self.saver = tf.train.Saver(max_to_keep=100)

            # debug graph:
            self.summary_writer.add_graph(self.session.graph)

    def broadcast(self, nonSpatialInput):
        nonSpatialInput = K.expand_dims(nonSpatialInput, axis=1)
        nonSpatialInput = K.expand_dims(nonSpatialInput, axis=2)
        return K.tile(nonSpatialInput,(1, self.ssize, self.ssize, 1))

    def expand_dims(self, x):
        return K.expand_dims(x, 1)

    def Squeeze(self, x):
        return K.squeeze(x,axis=-1)

    def build_net(self, dev):
         with tf.variable_scope('a3c') and tf.device(dev):
            screenInput = Input(
                shape=(U.screen_channel(), self.ssize, self.ssize),
                name='screenInput',
            )

            permutedScreenInput = Permute((2,3,1))(screenInput)
            conv1 = Conv2D(16, kernel_size=5, strides=(1,1), padding='same',name='conv1')(permutedScreenInput)
            conv2 = Conv2D(32, kernel_size=3, strides=(1,1), padding='same',name='conv2')(conv1)

            infoInput = Input(
                shape=(self.isize,),
                name='infoInput',
            )

            customInput = Input(
                shape=(self.custom_input_size,),
                name='customInput',
            )

            nonSpatialInput = Concatenate(name='nonSpatialInputConcat')([infoInput, customInput])

            broadcasted = Lambda(self.broadcast,name='broadcasting')(nonSpatialInput)

            combinedSpatialNonSpatial = Concatenate(name='combinedConcat')([broadcasted, conv2])

            conv3 = Conv2D(1, kernel_size=1, strides=(1,1), padding='same',name='conv3')(combinedSpatialNonSpatial)

            flatConv3 = Flatten(name='flatConv3')(conv3)

            lstmInput = Lambda(self.expand_dims, name='lstmInput')(flatConv3)

            self.NUM_LSTM = 100

            hStateInput = Input(
                shape=(self.NUM_LSTM,),
                name='hStateInput'
            )

            cStateInput = Input(
                shape=(self.NUM_LSTM,),
                name='cStateInput'
            )

            lstm, hStates, cStates = LSTM(self.NUM_LSTM, return_state=True)(lstmInput, initial_state=[hStateInput, cStateInput])

            fc1 = Dense(256, activation='relu',name='dense1')(lstm)
            fc2 = Dense(1, activation='linear',name='fc2')(fc1)
            value = Lambda(self.Squeeze,name='value')(fc2)
            policy = Dense(self.isize, activation='softmax',name='policy')(fc1)


            broadcastLstm = Lambda(self.broadcast, name='breadcastLstm')(lstm)

            spatialLstm = Concatenate(name='spatialLstm')([conv3, broadcastLstm])

            conv4 = Conv2D(1,kernel_size=1, strides=(1,1), padding='same',name='conv4')(spatialLstm)
            flatConv4 = Flatten(name='flattenedConv3')(conv4)
            spatialPolicy = Softmax(name='spatialPolicy')(flatConv4)

            conv5 = Conv2D(1, kernel_size=1, strides=(1,1), padding='same',name='conv5')(spatialLstm)
            flatConv5 = Flatten(name='flattenedConv5')(conv5)
            bestRoach = Softmax(name='bestRoach')(flatConv5)

            self.model = Model(
                inputs=[screenInput, infoInput, customInput, hStateInput, cStateInput],
                outputs=[value, policy, spatialPolicy, hStates, cStates, bestRoach]
            )
            self.model._make_predict_function()


    def save_model(self, path, count):
        self.saver.save(self.session, path + '/model.pkl', count)

    def load_model(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.session, ckpt.model_checkpoint_path)
        self.training_counter = int(ckpt.model_checkpoint_path.split('-')[-1])
