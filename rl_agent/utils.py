from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pysc2.lib import actions
from pysc2.lib import features

# TODO: preprocessing functions for the following layers
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SCREEN_SELECTED = features.SCREEN_FEATURES.selected.index

useful_screens = [
#4, # player_id
5, # player_relative
7, # selected
#8, # unit_hit_points
9, # unit_hit_points_ratio
#14, # unit_density
#15, # unit_density_aa
]
useful_actions = [
0, # no_op
#2, # select_point
#3, # select_rect
#4, # select_control_group
#5, # select_unit
#7, # select_army
12, # Attack_screen
#274, # HoldPosition_quick
331, # Move_screen
#333, # Patrol_screen
#453, # Stop_quick
]
compress_actions = {}
ii = 0
for x in useful_actions:
    compress_actions[x] = ii
    ii = ii + 1

def compressActions(ids):
  res = []
  for id in ids:
    if id in compress_actions:
      res.append(compress_actions[id])
  return res

def _isScalar(i):
  feat = features.SCREEN_FEATURES[i]
  return i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE or i == _SCREEN_SELECTED or feat.type == features.FeatureType.SCALAR

def preprocess_screen(screen):
  layers = []
  for i in useful_screens:
    feat = features.SCREEN_FEATURES[i]
    if _isScalar(i):
      layers.append(screen[i:i+1] / feat.scale)
    else:
      layer = np.zeros([feat.scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
      for j in range(feat.scale):
        indy, indx = (screen[i] == j).nonzero()
        layer[j, indy, indx] = 1
      layers.append(layer)
  return np.concatenate(layers, axis=0)



def screen_channel():
  c = 0
  for i in useful_screens:
    feat = features.SCREEN_FEATURES[i]
    if _isScalar(i):
      c += 1
    else:
      c += feat.scale
  return c
