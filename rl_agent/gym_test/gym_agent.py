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
import scipy.signal

frames = 0
EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 75000
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

    def getEpsilon(self):
        if(frames >= EPS_STEPS):
            return EPS_STOP
        else:
            return EPS_START + frames * (EPS_STOP - EPS_START) / EPS_STEPS	# linearly interpolate
    def act(self, observation):
        global frames; frames = frames + 1
        feed = self.brain.getPredictFeedDict(observation)
        action, value = self.brain.predict(feed)
        action = action.ravel()
        # Take the node highest with the most weight
        # action_id = np.argmax(action)
        action_id = np.random.choice(len(action), p=action)
        # if np.random.rand() < self.getEpsilon() and self.flags.training:
        #     action_id = np.random.choice(
        #         np.arange(len(action)))

        # Update status for GUI
        self.lastValue = value
        self.lastActionProbs = action
        one_hot = [int(i == action_id) for i in range(len(action))]
        return action_id, one_hot, value[0]


    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    # train_feed, p_feed, smask
    def train(self, last_observation, reward, action_onehot, value, observation, done):
        self.memory.append((reward, last_observation, action_onehot, value))
        brain = self.brain
        if len(self.memory) >= brain.N_STEP_RETURN or (done and len(self.memory) > 0):
            memory = self.memory
            r = 0
            v = 0
            if not done:
                _, v = brain.predict(brain.getPredictFeedDict(observation))
                v = v[0]
                r += v * brain.GAMMA
            batch = [[],[],[],[]]

            rewards = np.asarray([x[0] for x in memory])
            values = np.asarray([x[3] for x in memory] + [v])
            rewardsPlusV = np.append(rewards, [r])
            batch_r = self.discount(rewardsPlusV, brain.GAMMA)[:-1]
            delta_t = rewards + brain.GAMMA * values[1:] - values[:-1]
            batch_adv = self.discount(delta_t, brain.GAMMA * brain.LAMBDA)

            batch[0] = batch_r.tolist()
            batch[1] = [np.asarray(brain.preprocess(x[1])) for x in memory]
            batch[2] = [np.asarray(x[2]) for x in memory]
            batch[3] = batch_adv.tolist()

            brain.add_train(batch)
            self.memory = []
