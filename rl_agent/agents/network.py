from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers


def build_atari(screen, info, custom_input, ssize, num_action):
  # Extract features
  sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=8,
                         stride=4,
                         scope='sconv1')
  sconv2 = layers.conv2d(sconv1,
                         num_outputs=32,
                         kernel_size=4,
                         stride=2,
                         scope='sconv2')
  info_fc = layers.fully_connected(tf.concat([info, custom_input] ,axis=1),
                                   num_outputs=256,
                                   activation_fn=tf.tanh,
                                   scope='info_fc')

   # Compute spatial actions, non spatial actions and value
  feat_fc = tf.concat([layers.flatten(sconv2), info_fc], axis=1)
  feat_fc = layers.fully_connected(feat_fc,
                                   num_outputs=256,
                                   activation_fn=tf.nn.relu,
                                   scope='feat_fc')

  spatial_action_x = layers.fully_connected(feat_fc,
                                            num_outputs=ssize,
                                            activation_fn=tf.nn.softmax,
                                            scope='spatial_action_x')
  spatial_action_y = layers.fully_connected(feat_fc,
                                            num_outputs=ssize,
                                            activation_fn=tf.nn.softmax,
                                            scope='spatial_action_y')
  spatial_action_x = tf.reshape(spatial_action_x, [-1, 1, ssize])
  spatial_action_x = tf.tile(spatial_action_x, [1, ssize, 1])
  spatial_action_y = tf.reshape(spatial_action_y, [-1, ssize, 1])
  spatial_action_y = tf.tile(spatial_action_y, [1, 1, ssize])
  spatial_action = layers.flatten(spatial_action_x * spatial_action_y)

  # NOT BECAUSE WE ARE USING softmax_cross_entropy_with_logits WE SHOULD NOT
  # PREFORM SOFTMAX
  roach_x = layers.fully_connected(feat_fc,
                                            num_outputs=ssize,
                                            # activation_fn=tf.nn.softmax,
                                            scope='roach_x')
  roach_y = layers.fully_connected(feat_fc,
                                            num_outputs=ssize,
                                            # activation_fn=tf.nn.softmax,
                                            scope='roach_y')
  roach_x = tf.reshape(roach_x, [-1, 1, ssize])
  roach_x = tf.tile(roach_x, [1, ssize, 1])
  roach_y = tf.reshape(roach_y, [-1, ssize, 1])
  roach_y = tf.tile(roach_y, [1, 1, ssize])
  roach_prediction = layers.flatten(roach_x * roach_y)

  non_spatial_action = layers.fully_connected(feat_fc,
                                              num_outputs=num_action,
                                              activation_fn=tf.nn.softmax,
                                              scope='roach_prediction')
  value = tf.reshape(layers.fully_connected(feat_fc,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='value'), [-1])

  return spatial_action, non_spatial_action, value, roach_prediction

def build_net(screen, info, custom_input, ssize, num_action):
  # print(screen.shape)
  sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
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

  info_fc = layers.fully_connected(tf.concat([info,custom_input], axis=1),
                                   num_outputs=256,
                                   activation_fn=tf.nn.relu,
                                   scope='info_fc')

  # Compute spatial actions

  spatial_action = layers.conv2d(sconv2,
                                 num_outputs=1,
                                 kernel_size=1,
                                 stride=1,
                                 activation_fn=None,
                                 scope='spatial_action')
  spatial_action = tf.nn.softmax(layers.flatten(spatial_action))
  # print(sconv1.shape)
  # print(sconv2.shape)
  # print(spatial_action.shape)
  roach_prediction = layers.conv2d(sconv2,
                                 num_outputs=1,
                                 kernel_size=1,
                                 stride=1,
                                 activation_fn=None,
                                 scope='roach_prediction')
  roach_prediction = layers.flatten(roach_prediction)

  # Compute non spatial actions and value
  feat_fc = tf.concat([layers.flatten(sconv2), info_fc], axis=1)
  feat_fc = layers.fully_connected(feat_fc,
                                   num_outputs=int(256),
                                   activation_fn=tf.nn.relu,
                                   scope='feat_fc')
  non_spatial_action = layers.fully_connected(feat_fc,
                                              num_outputs=num_action,
                                              activation_fn=tf.nn.softmax,
                                              scope='non_spatial_action')
  value = tf.reshape(layers.fully_connected(feat_fc,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='value'), [-1])

  return spatial_action, non_spatial_action, value, roach_prediction
