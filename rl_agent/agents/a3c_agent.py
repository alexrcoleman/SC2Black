from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib import features

from agents.network import build_net
import utils as U


class A3CAgent(object):

  """An agent specifically for solving the mini-game maps."""
  def __init__(self, training, ssize,writer, graph_loss,  name='A3C/A3CAgent'):
    self.name = name
    self.training = training
    self.ssize = ssize
    self.isize = len(U.useful_actions)
    self.summary = []
    self.graph_loss = graph_loss

  def setup(self, sess, writer):
    self.sess = sess
    self.summary_writer = writer


  def initialize(self):
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)


  def getPolicyLoss(self, action_probability, advantage):
      return -tf.reduce_sum(tf.log(action_probability) * advantage)


  def getValueLoss(self, value_target, value_prediction):
      return tf.reduce_sum(tf.square(value_target - value_prediction))


  def getEntropy(self, policy, spatial_policy, valid_spatial):
      policy = tf.clip_by_value(policy, 1e-10, 1.)
      spatial_policy = tf.clip_by_value(spatial_policy, 1e-10, 1.)
      return -(tf.reduce_sum(policy * tf.log(policy)) + valid_spatial * tf.reduce_sum(spatial_policy * tf.log(spatial_policy)))


  def build_model(self, reuse, dev):
    with tf.variable_scope(self.name) and tf.device(dev):
      if reuse:
        tf.get_variable_scope().reuse_variables()
        assert tf.get_variable_scope().reuse

      # Set inputs of networks
      self.screen = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
      self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')

      # Build networks
      net = build_net(self.screen, self.info,  self.ssize, len(U.useful_actions))
      self.spatial_action, self.non_spatial_action, self.value = net

      # Set targets and masks
      self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
      self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize**2], name='spatial_action_selected')
      self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(U.useful_actions)], name='valid_non_spatial_action')
      self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(U.useful_actions)], name='non_spatial_action_selected')
      self.value_target = tf.placeholder(tf.float32, [None], name='value_target')
      # This will get the probability of choosing a valid action. Given that we force it to choose from
      # the set of valid actions. The probability of an action is the probability the policy chooses
      # divided by the probability of a valid action
      valid_non_spatial_action_probability = tf.reduce_sum(self.valid_non_spatial_action * self.non_spatial_action)
      non_spatial_action_probability = tf.reduce_sum(self.non_spatial_action_selected * self.non_spatial_action) / valid_non_spatial_action_probability

      # Here we compute the probability of the spatial action. If the action selected was non spactial,
      # the probability will be one.
      spatial_action_prob = (self.valid_spatial_action * tf.reduce_sum(self.spatial_action * self.spatial_action_selected) )+(1.0 - self.valid_spatial_action)

      # The probability of the action will be the the product of the non spatial and the spatial prob
      action_probability = non_spatial_action_probability * spatial_action_prob

      # The advantage function, which will represent how much better this action was than what was expected from this state
      advantage = self.value_target - self.value

      policy_loss = self.getPolicyLoss(action_probability, advantage)
      value_loss = self.getValueLoss(self.value_target, self.value)
      entropy = self.getEntropy(self.spatial_action, self.valid_spatial_action, self.spatial_action)


      loss =  policy_loss + value_loss *.5 + entropy * .01

      if self.graph_loss:
          self.summary.append(tf.summary.scalar('policy',tf.reduce_mean(policy_loss)))
          self.summary.append(tf.summary.scalar('value',tf.reduce_mean(value_loss)))
          self.summary.append(tf.summary.scalar('entropy',tf.reduce_mean(entropy)))
          self.summary.append(tf.summary.scalar('loss',tf.reduce_mean(loss)))
          self.summary_op = tf.summary.merge(self.summary)
          self.summary = []
      else:
          self.summary_op = []
      # Build the optimizer
      self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
      opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
      grads = opt.compute_gradients(loss)
      self.train_op = opt.apply_gradients(grads)

      self.saver = tf.train.Saver(max_to_keep=100)

  def step(self, obs):
    screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
    screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
    info = np.zeros([1, self.isize], dtype=np.float32)
    info[0, U.compressActions(obs.observation['available_actions'])] = 1

    feed = {
            self.screen: screen,
            self.info: info}
    non_spatial_action, spatial_action = self.sess.run(
      [self.non_spatial_action, self.spatial_action],
      feed_dict=feed)

    # Select an action and a spatial target
    non_spatial_action = non_spatial_action.ravel()
    spatial_action = spatial_action.ravel()
    valid_actions = obs.observation['available_actions']
    valid_actions = U.compressActions(valid_actions)

    net_act_id = valid_actions[np.random.choice(np.arange(len(non_spatial_action[valid_actions])), p=non_spatial_action[valid_actions]/np.sum(non_spatial_action[valid_actions]))]
    act_id = U.useful_actions[net_act_id]


    target = np.random.choice(np.arange(len(spatial_action)), p=spatial_action)

    target = [int(target // self.ssize), int(target % self.ssize)]

    act_args = []
    for arg in actions.FUNCTIONS[act_id].args:
      # set the location of the action
      if arg.name in ('screen', 'minimap', 'screen2'):
        act_args.append([target[1], target[0]])
      else:
        act_args.append([0])
    return actions.FunctionCall(act_id, act_args)


  def update(self, rbs, disc, lr, cter):
    # Compute R, which is value of the last observation
    obs = rbs[-1][-1]
    if obs.last():
      R = 0
    else:
      screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      info[0, U.compressActions(obs.observation['available_actions'])] = 1

      feed = {
              self.screen: screen,
              self.info: info}
      R = self.sess.run(self.value, feed_dict=feed)[0]

    # Compute targets and masks
    screens = []
    infos = []

    value_target = np.zeros([len(rbs)], dtype=np.float32)
    value_target[-1] = R

    valid_spatial_action = np.zeros([len(rbs)], dtype=np.float32)
    spatial_action_selected = np.zeros([len(rbs), self.ssize**2], dtype=np.float32)
    valid_non_spatial_action = np.zeros([len(rbs), len(U.useful_actions)], dtype=np.float32)
    non_spatial_action_selected = np.zeros([len(rbs), len(U.useful_actions)], dtype=np.float32)
    rbs.reverse()
    for i, [obs, action, next_obs] in enumerate(rbs):
      screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      info[0, U.compressActions(obs.observation['available_actions'])] = 1

      screens.append(screen)
      infos.append(info)

      reward = obs.reward
      act_id = action.function
      net_act_id = U.compress_actions[act_id]
      act_args = action.arguments


      player_relative = obs.observation.feature_screen.player_relative

      value_target[i] = reward + disc * value_target[i-1]

      valid_actions = obs.observation["available_actions"]
      valid_actions = U.compressActions(valid_actions)
      valid_non_spatial_action[i, valid_actions] = 1
      non_spatial_action_selected[i, net_act_id] = 1

      args = actions.FUNCTIONS[act_id].args
      for arg, act_arg in zip(args, act_args):
        if arg.name in ('screen', 'minimap', 'screen2'):
          ind = act_arg[1] * self.ssize + act_arg[0]
          valid_spatial_action[i] = 1
          spatial_action_selected[i, ind] = 1

    screens = np.concatenate(screens, axis=0)
    infos = np.concatenate(infos, axis=0)

    # Train
    feed = {
            self.screen: screens,
            self.info: infos,
            self.value_target: value_target,
            self.valid_spatial_action: valid_spatial_action,
            self.spatial_action_selected: spatial_action_selected,
            self.valid_non_spatial_action: valid_non_spatial_action,
            self.non_spatial_action_selected: non_spatial_action_selected,
            self.learning_rate: lr,
            }
    _, summary = self.sess.run([self.train_op, self.summary_op], feed_dict=feed)
    if self.graph_loss:
        self.summary_writer.add_summary(summary, cter)
        self.summary_writer.flush()

  def save_model(self, path, count):
    self.saver.save(self.sess, path+'/model.pkl', count)


  def load_model(self, path):
    ckpt = tf.train.get_checkpoint_state(path)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    return int(ckpt.model_checkpoint_path.split('-')[-1])
