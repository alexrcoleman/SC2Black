from gym_agent import A3CAgent
from pysc2.env import sc2_env
from pysc2.lib import actions
import numpy as np
import threading
import utils as U
import tensorflow as tf
import time
import gym

class Environment(threading.Thread):
    games = 0
    LOCK = threading.Lock()

    def __init__(self, flags, brain, summary_writer, name):
        threading.Thread.__init__(self)
        self.flags = flags
        self.brain = brain
        self.summary_writer = summary_writer
        self.stop_signal = False
        self.agent = A3CAgent(flags, brain, 64, name)
        self.lastTime = None

    def run(self):
        # Sets up environment
        self.env = gym.make(self.flags.environment)
        while not (self.stop_signal or self.brain.stop_signal):
            self.run_game()
    def run_game(self):
        FLAGS = self.flags
        num_frames = 0
        action_id = 0
        score = 0
        observation = self.env.reset()
        while not (self.stop_signal or self.brain.stop_signal):
            time.sleep(.00001) # yield
            num_frames += 1
            last_observation = observation
            if self.agent.name == "A3CAgent_0":
                self.env.render()
            action_id = self.agent.act(observation)
            observation, reward, done, _ = self.env.step(action_id)
            score += reward

            if FLAGS.training:
                self.agent.train(last_observation, reward, action_id, observation, done)

            if done:
                sum = tf.Summary()
                sum.value.add(tag='score', simple_value=score)
                with Environment.LOCK:
                    Environment.games = Environment.games + 1
                    local_games = Environment.games
                self.summary_writer.add_summary(sum, local_games)
                print('Game #' + str(local_games) + ' sample #' + str(self.brain.training_counter) + ' score: ' + str(score))
                break


    def stop(self):
        self.stop_signal = True
