from agent import A3CAgent
from pysc2.env import sc2_env
from pysc2.lib import actions
import numpy as np
import threading
import utils as U
import tensorflow as tf
import time


class Environment(threading.Thread):
    games = 0
    LOCK = threading.Lock()

    def __init__(self, flags, brain, max_frames, summary_writer, name):
        threading.Thread.__init__(self)
        self.flags = flags
        self.brain = brain
        self.summary_writer = summary_writer
        self.stop_signal = False
        self.agent = A3CAgent(flags, brain, 64, name)
        self.lastTime = None

    def startBench(self):
        self.lastTime = int(round(time.time() * 1000))
    def bench(self, label):
        millis = int(round(time.time() * 1000))
        print(self.agent.name, label, millis - self.lastTime, "ms")
        self.startBench()
    def run(self):

        # Sets up environment
        with sc2_env.SC2Env(map_name=self.flags.map,
                            agent_interface_format=agent_interface_format,
                            step_mul=self.flags.step_mul,
                            visualize=self.flags.render) as self.env:
            while not (self.stop_signal or self.brain.stop_signal):
                self.run_game()

    def run_game(self):
        FLAGS = self.flags
        num_frames = 0
        timestep = self.env.reset()[0]
        last_net_act_id = 0
        self.addInputs(timestep, num_frames, last_net_act_id)
        while not (self.stop_signal or self.brain.stop_signal):
            time.sleep(.00001) # yield
            num_frames += 1
            last_timestep = timestep

            last_net_act_id, act = self.agent.act(timestep)

            timestep = self.env.step([act])[0]
            self.addInputs(timestep, num_frames, last_net_act_id)

            is_done = (num_frames >= self.max_frames) or timestep.last()


            if FLAGS.training:
                self.agent.train(last_timestep, last_net_act_id, act, timestep)

            if is_done:
                score = timestep.observation["score_cumulative"][0]
                sum = tf.Summary()
                sum.value.add(tag='score', simple_value=score)
                with Environment.LOCK:
                    Environment.games = Environment.games + 1
                    local_games = Environment.games
                self.summary_writer.add_summary(sum, local_games)
                print('Game #' + str(local_games) + ' sample #' + str(self.brain.training_counter) + ' score: ' + str(score))
                break

    def addInputs(self, timestep, num_frames, last_net_act_id):
        norm_step = num_frames / self.max_frames
        last_act_onehot = np.zeros(
            [len(U.useful_actions)], dtype=np.float32)
        last_act_onehot[last_net_act_id] = 1
        timestep.observation.custom_inputs = np.concatenate(
            [[norm_step], last_act_onehot], axis=0)

    def stop(self):
        self.stop_signal = True
