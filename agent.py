from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from pysc2.lib import actions

import utils as U
import random
import copy
import scipy.signal

class A3CAgent(object):
    def __init__(self, flags, brain, ssize, name):
        self.name = name
        self.brain = brain
        self.flags = flags
        self.ssize = ssize
        self.isize = len(U.useful_actions)
        self.lastValue = -1
        self.lastValueTarget = -1
        self.lastLocation = [-1,-1]
        self.lastActionName = "None"
        self.lastActionProbs = "None"
        self.last_spatial = "None"
        self.R = 0
        self.memory = []

    def act(self, obs, hState, cState):
        feed = self.brain.getPredictFeedDict(obs, hState, cState)
        action_p, spatial_action_p, value, hS, cS = self.brain.predict(feed)

        # Select an action and a spatial target
        action_p = action_p.ravel()
        spatial_action_p = spatial_action_p.ravel()
        valid_actions = obs.observation['available_actions']
        valid_actions = U.compressActions(valid_actions)


        # Take the node highest with the most weight
        node_non_spatial_id = 0
        node_spatial_id = 0
        if len(valid_actions) > 0:
            valid_action_p = action_p[valid_actions] + 1e-12
            valid_action_p = valid_action_p / np.sum(valid_action_p)
            node_non_spatial_id = np.random.choice(np.arange(len(valid_actions)), p=valid_action_p)
            node_spatial_id = np.random.choice(np.arange(len(spatial_action_p)), p=spatial_action_p)

            # node_non_spatial_id = np.argmax(valid_action_p)
            # node_spatial_id = np.argmax(spatial_action_p)
            # if np.random.rand() < self.brain.eps and self.flags.training:
            #     node_non_spatial_id = np.random.choice(
            #         np.arange(len(valid_actions)))
            # if np.random.rand() < self.brain.eps and self.flags.training:
            #     node_spatial_id = np.random.choice(
            #         np.arange(len(spatial_action_p)))

        # Map these into actual actions / location

        action_one_hot = [int(i == node_non_spatial_id) for i in range(len(action_p))]
        spatial_one_hot = [int(i == node_spatial_id) for i in range(len(spatial_action_p))]

        net_act_id = 0
        if len(valid_actions) > 0:
            net_act_id = valid_actions[node_non_spatial_id]

        act_id = U.useful_actions[net_act_id]
        target = [int(node_spatial_id // self.ssize),
                  int(node_spatial_id % self.ssize)]

        if self.flags.force_focus_fire:
            roach_xy = U.getLowestHealthRoach(obs)
            dmarine_xy = U.getDangerousMarineLoc(obs)
            if act_id == 12:
                target = roach_xy
            if act_id == 2:
                target = dmarine_xy
                if target is None:
                    act_id = 0
        # Set the action arguments (currently only supports screen location)
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

        # Update status for GUI
        self.lastValue = value
        self.lastLocation = target
        self.lastActionName = actions.FUNCTIONS[act_id].name
        self.lastActionProbs = action_p
        self.last_spatial = spatial_action_p
        return net_act_id, actions.FunctionCall(act_id, act_args), action_one_hot, spatial_one_hot, value[0], hS[0], cS[0]

    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    # train_feed, p_feed, smask
    def train(self, timestep, action_onehot, spatial_onehot, value, next_timestep, act, act_id, last_hS, last_cS, hState, cState):
        brain = self.brain
        feature_dict = brain.getTrainFeedDict(timestep, act, act_id)
        self.memory.append((timestep.observation.rewardMod, feature_dict, action_onehot, spatial_onehot, value, last_hS, last_cS))
        if len(self.memory) >= brain.N_STEP_RETURN or (next_timestep.last() and len(self.memory) > 0):
            memory = self.memory
            r = 0
            v = 0
            if not next_timestep.last():
                _, _, v, _, _ = brain.predict(brain.getPredictFeedDict(next_timestep, hState, cState))
                v = v[0]
                r += v
            batch = [[],[],[],[],[],[],[]]


            rewards = np.asarray([x[0] for x in memory])
            values = np.asarray([x[4] for x in memory] + [v])
            # print("VALUES", values)
            rewardsPlusV = np.append(rewards, [r])
            batch_r = self.discount(rewardsPlusV, brain.GAMMA)[:-1]
            batch_r = self.discount(rewardsPlusV, brain.GAMMA)[:-1]
            delta_t = rewards + brain.GAMMA * values[1:] - values[:-1]
            batch_adv = self.discount(delta_t, brain.GAMMA * brain.LAMBDA)

            batch[0] = batch_r.tolist()
            batch[1] = [x[1] for x in memory]
            batch[2] = [np.asarray(x[2]) for x in memory]
            batch[3] = [np.asarray(x[3]) for x in memory]
            batch[4] = batch_adv.tolist()
            batch[5] = [np.asarray(x[5]) for x in memory]
            batch[6] = [np.asarray(x[6]) for x in memory]

            brain.add_train(batch)
            self.memory = []
