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
        action_id = np.argmax(action)
        # action_id = np.random.choice(len(action), p=action)
        if np.random.rand() < self.getEpsilon() and self.flags.training:
            action_id = np.random.choice(
                np.arange(len(action)))

        # Update status for GUI
        self.lastValue = value 
        self.lastActionProbs = action
        one_hot = [int(i == action_id) for i in range(len(action))]
        return action_id, one_hot

    # train_feed, p_feed, smask
    def train(self, last_observation, reward, action_id, observation, done):
        self.memory.append((reward, last_observation, action_id))
        brain = self.brain
        # print("BEG")
        if len(self.memory) >= brain.N_STEP_RETURN or (done and len(self.memory) > 0):
            r = 0
            if not done:
                _, v = brain.predict(brain.getPredictFeedDict(observation))
                r += v[0]
            # print("predict", v)
            batch = [[],[],[]]
            for sample in self.memory[::-1]:
                # print("REWARD ", sample[0], " RUNNING ", r)
                # print("ACTION", action)
                r += sample[0]
                r *= brain.GAMMA
                batch[0].append(r)
                batch[1].append(np.array(brain.preprocess(sample[1]), dtype=np.float32))
                batch[2].append(sample[2])
            # print(batch[0], "END")
            brain.add_train(batch)
            self.memory = []
