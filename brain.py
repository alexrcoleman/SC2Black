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
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv2DTranspose
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
        self.train_queue = [[], [], [], [], [], [], []]


        self.counter_lock = threading.Lock()
        self.training_counter = 0
        tf.reset_default_graph()
        K.clear_session()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        K.set_session(self.session)
        K.manual_variable_initialization(True)
        self.build_net('/gpu:0')
        self.default_graph = tf.get_default_graph()
        self.summary_writer.add_graph(self.default_graph)
        self.build_model('/gpu:0')
        self.session.run(tf.global_variables_initializer())
        self.default_graph.finalize()
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
            self.cStateInput: cState,
            self.xB:np.array(U.makeX(len(screen),self.ssize)),
            self.yB:np.array(U.makeY(len(screen),self.ssize)),
            }

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
            v, policy, spatialPolicy, hS, cS  = self.model.predict([
                feed[self.screen],
                feed[self.info],
                feed[self.custom_inputs],
                feed[self.xB],
                feed[self.yB],
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
            self.train_queue = [[],[],[],[],[],[],[]]

        batch_train_feed = {
            self.value_target:np.squeeze(np.array(batch[0], dtype=np.float32)),
            self.screen:np.asarray([x[self.screen] for x in batch[1]],dtype=np.float32),
            self.info:np.asarray([x[self.info] for x in batch[1]],dtype=np.float32),
            self.custom_inputs:np.asarray([x[self.custom_inputs] for x in batch[1]],dtype=np.float32),
            self.valid_spatial_action:np.asarray([x[self.valid_spatial_action] for x in batch[1]],dtype=np.float32),
            self.valid_action:np.asarray([x[self.valid_action] for x in batch[1]],dtype=np.float32),
            self.action_selected:np.array(batch[2], dtype=np.float32),
            self.spatial_action_selected:np.array(batch[3], dtype=np.float32),
            self.hStateInput:np.array(batch[5], dtype=np.float32),
            self.cStateInput:np.array(batch[6], dtype=np.float32),
            self.advantage:np.array(batch[4], dtype=np.float32),
            self.xB:np.array(U.makeX(len(batch[0]), self.ssize)),
            self.yB:np.array(U.makeY(len(batch[0]), self.ssize)),
        }

        self.train(batch_train_feed)

    def getPolicyLoss(self, action_probability, advantage):
        return -tf.log(action_probability + 1e-10) * advantage

    def getValueLoss(self, difference):
        return tf.square(difference)

    def getEntropy(self, policy, spatial_policy, valid_spatial):
        return tf.reduce_sum(policy * tf.log(policy + 1e-10), axis=1)# + valid_spatial * tf.reduce_sum(spatial_policy * tf.log(spatial_policy + 1e-10), axis=1)

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
                tf.float32, [None,] + [self.state_shape[i] for i in range(len(self.state_shape))], name='h_state_input')
            self.cStateInput = tf.placeholder(
                tf.float32, [None,] + [self.state_shape[i] for i in range(len(self.state_shape))], name='c_state_input')
            self.xB = tf.placeholder(
                tf.float32, [None, 8, 8 ,1], name='xB'
            )
            self.yB = tf.placeholder(
                tf.float32, [None, 8, 8 ,1], name='yB'
            )
            # print(self.screen.shape, self.info.shape, self.custom_inputs.shape, self.hStateInput.shape, self.cStateInput.shape, self.xB.shape, self.yB.shape)
            self.value, self.policy, self.spatial_policy, _, _ = self.model([self.screen, self.info, self.custom_inputs,self.xB,self.yB, self.hStateInput, self.cStateInput])
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


            loss = tf.reduce_mean(policy_loss + value_loss * .5 + entropy * .01)
            # Build the optimizer
            global_step = tf.Variable(0)
            learning_rate_decayed = tf.train.exponential_decay(self.learning_rate,
                                                               global_step,10000, .95)

            opt = tf.train.AdamOptimizer(self.learning_rate)
            grads, vars = zip(*opt.compute_gradients(loss))
            grads, glob_norm = tf.clip_by_global_norm(grads, 5.0)
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
            self.summary_writer.add_graph(self.session.graph)

    def splitQKV(self,x):
        x = Permute((2,1,3))(x)
        q,k,v = tf.split(x, [self.key_size, self.key_size, self.head_size], -1)
        q *= self.key_size ** -.05
        return [q, k, v]

    def conv2DLSTM(self, inputs, state, output_channels, kernel_size, forgetBias = 1.0):
        hidden, cell = state
        inputConv = Conv2D(4 * output_channels, kernel_size = kernel_size, strides=(1,1), padding='same')(inputs)
        hiddenConv = Conv2D(4 * output_channels, kernel_size = kernel_size, strides=(1,1), padding='same')(hidden)
        nextHidden = Add()([inputConv, hiddenConv])

        gates = tf.split(value = nextHidden, num_or_size_splits=4, axis=3)
        inputGate, nextInput, forgetGate, outputGate = gates
        nextCell = tf.sigmoid(forgetGate + forgetBias) * cell
        nextCell += tf.sigmoid(inputGate) * tf.tanh(nextInput)
        output = tf.tanh(nextCell) * tf.sigmoid(outputGate)

        return [output, output, nextCell]


    def build_net(self,dev):
        with tf.variable_scope('a3c') and tf.device(dev):
           screenInput = Input(
               shape=(U.screen_channel(), self.ssize, self.ssize),
               name='screenInput',
           )

           input3D = Permute((2,3,1))(screenInput)
           for i in range(2):
               input3D = Conv2D(32, kernel_size=4, strides=(2,2), padding='same')(input3D)
               input3D = Conv2D(64, kernel_size=3, strides=(1,1), padding='same')(input3D)
               input3D = Conv2D(94, kernel_size=3, strides=(1,1), padding='same')(input3D)

           infoInput = Input(
               shape=(self.isize,),
               name='infoInput',
           )

           customInput = Input(
               shape=(self.custom_input_size,),
               name='customInput',
           )

           input2D = Concatenate(name='nonSpatialInputConcat')([infoInput, customInput])
           input2D = Dense(128, activation='relu')(input2D)
           input2D = Dense(64, activation='relu')(input2D)

           broadcasted = Lambda(K.expand_dims,arguments={'axis':1})(input2D)
           broadcasted = Lambda(K.expand_dims,arguments={'axis':2})(broadcasted)
           broadcasted = Lambda(K.tile,arguments={'n':(1,8,8,1)})(broadcasted)

           combinedSpatialNonSpatial = Concatenate(name='combinedConcat')([broadcasted, input3D])
           self.state_shape = (8,8,96)

           hStateInput = Input(
               shape=self.state_shape,
               name='hStateInput'
           )

           cStateInput = Input(
               shape=self.state_shape,
               name='cStateInput'
           )


           output2d, hState, cState= Lambda(self.conv2DLSTM, arguments={
             'state':(hStateInput, cStateInput),
             'output_channels':96,
             'kernel_size':4,
           })(combinedSpatialNonSpatial)


           xB = Input(shape=(8,8,1), name='xb')
           yB = Input(shape=(8,8,1), name='yb')

           coordLstm = Concatenate(name='coordLstm')([output2d, xB, yB])

           self.num_heads = 2
           self.head_size = 32
           self.key_size = 32

           # unravel the width dimension new shape = [None, 32^2, depth]
           unraveled = Reshape((8 * 8, 98))(coordLstm)
           self.qkv_size = self.head_size + self.key_size * 2
           self.total_size = self.qkv_size * self.num_heads
           qkv = Conv1D(self.total_size, kernel_size=1, strides=(1,), activation='linear')(unraveled)
           qkv = BatchNormalization()(qkv)
           qkv = Reshape((8*8,self.num_heads,self.qkv_size))(qkv)

           q, k, v = Lambda(self.splitQKV)(qkv)

           dot = Lambda(tf.matmul, arguments={'b':k,'transpose_b':True})(q)
           weights = Softmax()(dot)
           output = Lambda(tf.matmul, arguments={'b':v})(weights)

           output_t = Permute((2,1,3))(output)

           attention = Reshape((8*8,output_t.shape[2] * output_t.shape[3]))(output_t)
           attention = Conv1D(self.state_shape[2] + 2, kernel_size=1, strides=(1,), activation='relu')(attention)
           attention = Conv1D(self.state_shape[2] + 2, kernel_size=1, strides=(1,), activation='relu')(attention)
           attention = Reshape((8,8,self.state_shape[2]+2))(attention)
           final = Add()([attention, coordLstm])

           nonSpatial = MaxPool2D(pool_size=(8,8))(final)
           nonSpatial = Reshape((98,))(nonSpatial)
           nonSpatial = Concatenate()([nonSpatial, input2D])


           vfc = Dense(256, activation='relu',name='vfc')(nonSpatial)
           vfc = Dense(1, activation='linear',name='fc2')(vfc)
           value = Lambda(K.squeeze,arguments={'axis':-1},name='value')(vfc)

           pfc = Dense(256, activation='relu',name='pfc')(nonSpatial)
           policy_logits = Dense(self.isize ,name='policy')(pfc)
           policy = Softmax()(policy_logits)


           spatialOut = Conv2DTranspose(32,kernel_size=4,strides=(2,2),padding='same')(final)
           spatialOut = Conv2DTranspose(16,kernel_size=4,strides=(2,2),padding='same')(spatialOut)

           broadcastPolicy = Lambda(K.expand_dims,arguments={'axis':1})(policy_logits)
           broadcastPolicy = Lambda(K.expand_dims,arguments={'axis':2})(broadcastPolicy)
           broadcastPolicy = Lambda(K.tile,arguments={'n':(1,32,32,1)})(broadcastPolicy)

           spatialOut = Concatenate()([spatialOut, broadcastPolicy])

           spatialPolicy = Conv2D(1,kernel_size=1, strides=(1,1), padding='same',name='conv4')(spatialOut)
           spatialPolicy = Flatten(name='flattenedConv3')(spatialPolicy)
           spatialPolicy = Softmax(name='spatialPolicy')(spatialPolicy)

           self.model = Model(
               inputs=[screenInput, infoInput, customInput,xB, yB,  hStateInput, cStateInput],
               outputs=[value, policy, spatialPolicy, hState, cState]
           )
           self.model._make_predict_function()

    def save_model(self, path, count):
        self.saver.save(self.session, path + '/model.pkl', count)

    def load_model(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.session, ckpt.model_checkpoint_path)
        self.training_counter = int(ckpt.model_checkpoint_path.split('-')[-1])
