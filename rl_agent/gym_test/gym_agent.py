from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from pysc2.lib import actions

import utils as U
import random


class A3CAgent(object):
    def __init__(self, flags, brain, ssize, name):
        self.name = name
        self.brain = brain
        self.flags = flags
        self.lastValue = -1
        self.lastValueTarget = -1
        self.lastActionName = "None"
        self.lastActionProbs = "None"
        self.R = 0
        self.memory = []

    def act(self, observation):
        feed = self.brain.getPredictFeedDict(observation)
        action, value = self.brain.predict(feed)

        # Take the node highest with the most weight
        action_id = 0
        action_id = np.argmax(action)
        if np.random.rand() < self.brain.eps and self.flags.training:
            action_id = np.random.choice(
                np.arange(len(action)))

        # Update status for GUI
        self.lastValue = value
        self.lastActionProbs = action
        return action_id

    # train_feed, p_feed, smask
    def train(self, last_observation, reward, act_id, observation, done):
        brain = self.brain
        def get_sample(memory):
            tf, _, _ = memory[0]
            _, pf, smask = memory[-1]
            memory.pop(0)
            return tf, pf, self.R, smask

        # print("Self memory: ", len(self.memory), self.R)
        self.R = (self.R + reward * brain.GAMMA_N) / brain.GAMMA

        self.memory.append((brain.getTrainFeedDict(timestep, act, net_act_id), brain.getPredictFeedDict(
            next_timestep), 0 if done else 1))

        # in case the game ended in < N steps (shouldn't happen)
        if done:
            self.R = self.R / brain.GAMMA**(brain.N_STEP_RETURN-len(self.memory))
        while len(self.memory) >= brain.N_STEP_RETURN or (done and len(self.memory) > 0):
            tf, pf, r, smask = get_sample(self.memory)
            self.R = self.R - tf[brain.value_target][0]
            if done:
                self.R = self.R / brain.GAMMA
            tf[brain.value_target] = np.array([r])
            brain.add_train(tf, pf, smask)
        if done:
            self.R = 0
