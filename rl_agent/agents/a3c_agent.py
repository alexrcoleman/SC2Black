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

# For computing new rewards:
_PLAYER_SELF = features.PlayerRelative.SELF
def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))

class A3CAgent(object):


  """An agent specifically for solving the mini-game maps."""
  def __init__(self, training, ssize, force_focus_fire, name='A3C/A3CAgent'):
    self.name = name
    self.training = training
    self.summary = []
    self.force_focus_fire = force_focus_fire
    # Screen size and info size
    self.ssize = ssize
    self.isize = len(U.useful_actions)
    self.epsilon = [0.05, 0.2]


  def setup(self, sess, summary_writer):
    self.sess = sess
    self.summary_writer = summary_writer


  def initialize(self):
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)


  def reset(self):
    if False:
      # Epsilon schedule
      self.epsilon = [0.05, 0.2]

  def build_model(self, reuse, dev, ntype):
    with tf.variable_scope(self.name) and tf.device(dev):
      if reuse:
        tf.get_variable_scope().reuse_variables()
        assert tf.get_variable_scope().reuse

      # Set inputs of networks
      self.screen = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
      self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')
      # todo: actually put some inputs in this vec
      self.extra_input = tf.placeholder(tf.float32, [None, 3], name='info')

      # Build networks
      net = build_net(self.screen, self.info, self.extra_input, self.ssize, len(U.useful_actions) , ntype)
      self.spatial_action, self.non_spatial_action, self.value = net

      # Set targets and masks
      self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
      self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize**2], name='spatial_action_selected')
      self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(U.useful_actions)], name='valid_non_spatial_action')
      self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(U.useful_actions)], name='non_spatial_action_selected')
      self.value_target = tf.placeholder(tf.float32, [None], name='value_target')

      # Compute log probability
      spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
      spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
      non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
      valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action, axis=1)
      valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
      non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
      non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
      self.summary.append(tf.summary.histogram('spatial_action_prob', spatial_action_prob))
      self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))

      # Compute losses, more details in https://arxiv.org/abs/1602.01783
      # Policy loss and value loss
      action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
      advantage = tf.stop_gradient(self.value_target - self.value)
      policy_loss = - tf.reduce_mean(action_log_prob * advantage)
      value_loss = - tf.reduce_mean(self.value * advantage)
      self.summary.append(tf.summary.scalar('policy_loss', policy_loss))
      self.summary.append(tf.summary.scalar('value_loss', value_loss))

      # TODO: policy penalty
      loss = policy_loss + value_loss

      # Build the optimizer
      self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
      opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
      grads = opt.compute_gradients(loss)
      cliped_grad = []
      for grad, var in grads:
        self.summary.append(tf.summary.histogram(var.op.name, var))
        self.summary.append(tf.summary.histogram(var.op.name+'/grad', grad))
        grad = tf.clip_by_norm(grad, 10.0)
        cliped_grad.append([grad, var])
      self.train_op = opt.apply_gradients(cliped_grad)
      self.summary_op = tf.summary.merge(self.summary)

      self.saver = tf.train.Saver(max_to_keep=100)

  def _xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))

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
    # print(valid_actions)
    og_id = np.argmax(non_spatial_action)
    net_act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
    # print(valid_actions)
    # print(non_spatial_action)
    # print(net_act_id)
    act_id = U.useful_actions[net_act_id]
    if act_id == 1000:
        act_id = 5
        marines = [unit for unit in obs.observation.feature_units if unit.alliance != features.PlayerRelative.ENEMY]
        best_marine = 0
        minHealth = 1000000
        minDist = 0
        for marine in marines:
          hp = marine.health
          if hp < minHealth:
            minHealth = hp
            best_marine = marine
        spatial_action = best_marine.y * self.ssize + best_marine.x

    elif act_id == 1001:
        act_id = 12
        # print("GERE")
        # get the average marine location
        player_relative = obs.observation.feature_screen.player_relative
        marines = _xy_locs(player_relative == _PLAYER_SELF)
        marine_xy = np.mean(marines, axis=0).round()

        # Find the nearest low roach and save it so we know to attack with it
        roaches = [unit for unit in obs.observation.feature_units if unit.alliance == features.PlayerRelative.ENEMY]
        best_roach = 0
        minHealth = 1000000
        minDist = 0
        for roach in roaches:
          hp = roach.health
          dist = np.sqrt((marine_xy[0]-roach.x)**2 + (marine_xy[1]-roach.y)**2)
          if hp < minHealth or (hp == minHealth and dist < minDist):
            minHealth = hp
            minDist = dist
            best_roach = roach
        #roach_xy = [best_roach.y, best_roach.x]
        spatial_action = best_roach.y * self.ssize + best_roach.x

    target = np.argmax(spatial_action)
    target = [int(target // self.ssize), int(target % self.ssize)]

    # show the action being chosen:
    #  print(actions.FUNCTIONS[act_id].name, target)

    # Epsilon greedy exploration
    if self.training and np.random.rand() < self.epsilon[0]:
      net_act_id = np.random.choice(valid_actions)
      act_id = U.useful_actions[net_act_id]
      if act_id == 1000:
          act_id = 5
          marines = [unit for unit in obs.observation.feature_units if unit.alliance != features.PlayerRelative.ENEMY]
          best_marine = 0
          minHealth = 1000000
          minDist = 0
          for marine in marines:
            hp = marine.health
            if hp < minHealth:
              minHealth = hp
              best_marine = marine
          target = [best_marine.y, best_marine.x]
        #  spatial_action = best_marine.y * self.ssize + best_marine.x

      elif act_id == 1001:
          act_id = 12
          # print("GERE")
          # get the average marine location
          player_relative = obs.observation.feature_screen.player_relative
          marines = _xy_locs(player_relative == _PLAYER_SELF)
          marine_xy = np.mean(marines, axis=0).round()

          # Find the nearest low roach and save it so we know to attack with it
          roaches = [unit for unit in obs.observation.feature_units if unit.alliance == features.PlayerRelative.ENEMY]
          best_roach = 0
          minHealth = 1000000
          minDist = 0
          for roach in roaches:
            hp = roach.health
            dist = np.sqrt((marine_xy[0]-roach.x)**2 + (marine_xy[1]-roach.y)**2)
            if hp < minHealth or (hp == minHealth and dist < minDist):
              minHealth = hp
              minDist = dist
              best_roach = roach
          target = [best_roach.y, best_roach.x]
          #spatial_action = best_roach.y * self.ssize + best_roach.x
    if self.training and np.random.rand() < self.epsilon[0]:
        target[0] = np.random.randint(0, self.ssize)
        target[1] = np.random.randint(0, self.ssize)
    if self.training and np.random.rand() < self.epsilon[1]:
      dy = np.random.randint(-6, 7)
      target[0] = int(max(0, min(self.ssize-1, target[0]+dy)))
      dx = np.random.randint(-6, 7)
      target[1] = int(max(0, min(self.ssize-1, target[1]+dx)))

    #print(actions.FUNCTIONS[act_id].name, target)
    act_args = []
    for arg in actions.FUNCTIONS[act_id].args:
      # set the location of the action
      if arg.name in ('screen', 'minimap', 'screen2'):
        act_args.append([target[1], target[0]])
      else:
        # TODO: Handle this better with more output
        # ex. control groups requires specifying both the control group
        # id and whether you are selecting or setting the group
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
      marines = _xy_locs(player_relative == _PLAYER_SELF)
      # ex. reward based on distance
      if len(marines) >= 2 and False:
        bonus = 0
        for i in range(0,len(marines)):
          for j in range(i+1,len(marines)):
            bonus = bonus + np.linalg.norm(np.array(marines[i])-np.array(marines[j]))
        bonus = bonus * .001
        if reward > 0:
          # scale by (1+bonus) so each shard collected rewards them 1 point +
          # distance apart
          reward = reward * (1 + bonus)

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
            self.learning_rate: lr}
    _, summary = self.sess.run([self.train_op, self.summary_op], feed_dict=feed)
    self.summary_writer.add_summary(summary, cter)


  def save_model(self, path, count):
    self.saver.save(self.sess, path+'/model.pkl', count)


  def load_model(self, path):
    ckpt = tf.train.get_checkpoint_state(path)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    return int(ckpt.model_checkpoint_path.split('-')[-1])
