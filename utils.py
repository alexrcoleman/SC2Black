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
    # 4, # player_id
    5,  # player_relative
    # 7, # selected
    # 8, # unit_hit_points
    9,  # unit_hit_points_ratio
    # 14, # unit_density
    # 15, # unit_density_aa
]
useful_actions = [
    0,  # no_op
    2,  # select_point
    # 3, # select_rect
    # 4, # select_control_group
    # 5, # select_unit
    # ^ this let's you adjust something in your selection; could be useful because
    # your agent could look at the health of units from multiselection from that special feature
    # and grab the lowest hp one; since we don't have that though, its off
    7,  # select_army
    12,  # Attack_screen
    # 274, # HoldPosition_quick
    331,  # Move_screen
    # 333, # Patrol_screen
    # 453, # Stop_quick
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

def xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))

# Find the lowest health marine "in danger"
def getDangerousMarineLoc(obs):
    feature_marines = [
        unit for unit in obs.observation.feature_units if unit.alliance == features.PlayerRelative.SELF]
    best_marine = None
    best_roach = getLowestHealthRoach(obs)
    if best_roach is None:
        return None
    minHealth = 1000000
    for marine in feature_marines:
        hp = marine.health_ratio / 255
        dist = np.sqrt((marine.x - best_roach[1])**2 + (marine.y - best_roach[0])**2)
        if hp < .5 and dist < 15:
            if hp < minHealth:
                best_marine = marine
                minHealth = hp
    if not best_marine is None:
        best_marine = [best_marine.y, best_marine.x]
    return best_marine

def getMarineHealthSum(obs):
    sumHealth = 0
    feature_marines = [
          unit for unit in obs.observation.feature_units if unit.alliance == features.PlayerRelative.SELF]   
    for marine in feature_marines:
        hp = marine.health_ratio / 255
        sumHealth = sumHealth + hp
    return sumHealth
        
def make_batch(feeds):
    feed = {}
    for key in feeds[0]:
        feed[key] = np.copy(feeds[0][key])
    for ofeed in feeds[1:]:
        for key in ofeed:
            feed[key] = np.append(feed[key], ofeed[key], axis=0)
    return feed

# Find the nearest low roach and save it so we know to attack with it
def getLowestHealthRoach(obs):
    player_relative = obs.observation.feature_screen.player_relative
    marines = xy_locs(player_relative == features.PlayerRelative.SELF)
    marine_xy = np.mean(marines, axis=0).round()

    roaches = [unit for unit in obs.observation.feature_units if unit.alliance ==
               features.PlayerRelative.ENEMY]
    best_roach = None
    minHealth = 1000000
    minDist = 0
    for roach in roaches:
        hp = roach.health
        dist = np.sqrt((marine_xy[0] - roach.x)**2 + (marine_xy[1] - roach.y)**2)
        if hp < minHealth or (hp == minHealth and dist < minDist):
            minHealth = hp
            minDist = dist
            best_roach = roach
    if not best_roach is None:
        best_roach = [best_roach.y, best_roach.x]
    return best_roach


use_coords = True
def preprocess_screen(screen):
    layers = []
    for i in useful_screens:
        feat = features.SCREEN_FEATURES[i]
        if _isScalar(i):
            layers.append(screen[i:i + 1] / feat.scale)
        else:
            layer = np.zeros([feat.scale, screen.shape[1],
                              screen.shape[2]], dtype=np.float32)
            for j in range(feat.scale):
                indy, indx = (screen[i] == j).nonzero()
                layer[j, indy, indx] = 1
            layers.append(layer)
    if use_coords:
        x = np.zeros([screen.shape[1], screen.shape[2]], dtype=np.float32)
        y = np.zeros([screen.shape[1], screen.shape[2]], dtype=np.float32)
        for i in range(screen.shape[1]):
            for j in range(screen.shape[2]):
                x[i][j] = i / screen.shape[1]
                y[i][j] = j / screen.shape[2]
        layers.append([x, y])

    return np.concatenate(layers, axis=0)

def screen_channel():
    c = 0
    if use_coords:
        c += 2
    for i in useful_screens:
        feat = features.SCREEN_FEATURES[i]
        if _isScalar(i):
            c += 1
        else:
            c += feat.scale
    return c

def runningAverage(list, size):
    cnt = 0
    sum = 0
    nlist = []
    for i in range(len(list)):
        sum += list[i]
        if i >= size:
            sum -= list[i - size]
        if i + 1 >= size:
            nlist.append(sum / size)
    return nlist
