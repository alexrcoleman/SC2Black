from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from pysc2.lib import actions

import utils as U
import random
import copy

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

    def act(self, obs):
        feed = self.brain.getPredictFeedDict(obs)
        non_spatial_action_p, spatial_action_p, value = self.brain.predict(feed)

        # Select an action and a spatial target
        non_spatial_action_p = non_spatial_action_p.ravel()
        old = spatial_action_p
        spatial_action_p = spatial_action_p.ravel()
        valid_actions = obs.observation['available_actions']
        valid_actions = U.compressActions(valid_actions)

        # Take the node highest with the most weight
        node_non_spatial_id = 0
        node_spatial_id = 0
        if len(valid_actions) > 0:
            valid_action_p = non_spatial_action_p[valid_actions] + 1e-12
            valid_action_p = valid_action_p / np.sum(valid_action_p)
            # node_non_spatial_id = np.random.choice(np.arange(len(valid_actions)), p=valid_action_p)
            # node_spatial_id = np.random.choice(np.arange(len(spatial_action_p)), p=spatial_action_p)
            node_non_spatial_id = np.argmax(valid_action_p)
            node_spatial_id = np.argmax(spatial_action_p)
            if np.random.rand() < self.brain.eps and self.flags.training:
                node_non_spatial_id = np.random.choice(
                    np.arange(len(valid_actions)))
            if np.random.rand() < self.brain.eps and self.flags.training:
                node_spatial_id = np.random.choice(
                    np.arange(len(spatial_action_p)))
                    
        # Map these into actual actions / location
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
        self.lastActionProbs = non_spatial_action_p
        self.last_spatial = spatial_action_p
        return net_act_id, actions.FunctionCall(act_id, act_args)

    # train_feed, p_feed, smask
    def train(self, timestep, net_act_id, act, next_timestep):
        brain = self.brain
        def get_sample(memory):
            tf, _, _ = memory[0]
            _, pf, smask = memory[-1]
            memory.pop(0)
            return tf, copy.copy(pf), self.R, smask

        # print("Self memory: ", len(self.memory), self.R)
        self.R = (self.R + timestep.reward * brain.GAMMA_N) / brain.GAMMA

        self.memory.append((brain.getTrainFeedDict(timestep, act, net_act_id), brain.getPredictFeedDict(
            next_timestep), 0 if next_timestep.last() else 1))

        # in case the game ended in < N steps (shouldn't happen)
        if timestep.last():
            self.R = self.R / brain.GAMMA**(brain.N_STEP_RETURN-len(self.memory))
            print("WARNING: Game ended in under %d steps" % brain.N_STEP_RETURN)
        while len(self.memory) >= brain.N_STEP_RETURN or (next_timestep.last() and len(self.memory) > 0):
            tf, pf, r, smask = get_sample(self.memory)
            self.R = self.R - tf[brain.value_target][0]
            if next_timestep.last():
                self.R = self.R / brain.GAMMA
            tf[brain.value_target] = np.array([r])
            brain.add_train(tf, pf, smask)
        if next_timestep.last():
            self.R = 0
