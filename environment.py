from agent import A3CAgent
from pysc2.env import sc2_env
from pysc2.lib import actions
import numpy as np
import threading
import utils as U
import tensorflow as tf
import time
from pysc2.lib import features


class Environment(threading.Thread):
    games = 0
    LOCK = threading.Lock()

    def __init__(self, flags, brain, max_frames, summary_writer, name):
        threading.Thread.__init__(self)
        self.flags = flags
        self.brain = brain
        self.max_frames = max_frames
        self.summary_writer = summary_writer
        self.stop_signal = False
        self.agent = A3CAgent(flags, brain, 32, name)
        self.lastTime = None

    def startBench(self):
        self.lastTime = int(round(time.time() * 1000))
    def bench(self, label):
        millis = int(round(time.time() * 1000))
        print(self.agent.name, label, millis - self.lastTime, "ms")
        self.startBench()
    def run(self):
        # Sets up interaction with environment
        agent_interface_format = sc2_env.parse_agent_interface_format(
            feature_screen=32,
            feature_minimap=16,
            use_feature_units=True)

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
        hState = np.zeros(self.brain.NUM_LSTM)
        cState = np.zeros(self.brain.NUM_LSTM)
        while not (self.stop_signal or self.brain.stop_signal):
            time.sleep(.00001) # yield
            #self.startBench()
            num_frames += 1
            last_timestep = timestep

            if FLAGS.force_focus_fire:
                dmarine = U.getDangerousMarineLoc(timestep)
                if dmarine is None:
                    index = np.argwhere(
                        timestep.observation.available_actions == 2)
                    timestep.observation.available_actions = np.delete(
                        timestep.observation.available_actions, index)
            #self.bench("part1")
            last_hS = hState
            last_cS = cState
            last_net_act_id, act, action_onehot, spatial_onehot, value, hState, cState = self.agent.act(timestep,hState,cState)
            #self.bench("agent")
            timestep = self.env.step([act])[0]
            #self.bench("env a")
            self.addInputs(timestep, num_frames, last_net_act_id)

            is_done = (num_frames >= self.max_frames) or timestep.last()


            if FLAGS.training:
                self.agent.train(last_timestep, action_onehot, spatial_onehot, value, timestep, act, last_net_act_id, last_hS, last_cS, hState, cState)
            #self.bench("train")
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
        FLAGS = self.flags
        norm_step = num_frames / self.max_frames
        last_act_onehot = np.zeros(
            [len(U.useful_actions)], dtype=np.float32)
        last_act_onehot[last_net_act_id] = 1
        timestep.observation.custom_inputs = np.concatenate(
            [[norm_step], last_act_onehot], axis=0)
        if FLAGS.kiting:
            kiteReward = U.KiteEnemies(timestep)
            timestep.observation.rewardMod = timestep.reward + kiteReward
            marines = [
                unit for unit in timestep.observation.feature_units if unit.alliance == features.PlayerRelative.SELF]
            roaches = [unit for unit in timestep.observation.feature_units if unit.alliance ==
                features.PlayerRelative.ENEMY]
            for marine in marines:
                for roach in roaches:
                    dist = np.sqrt((marine.x - roach.x)**2 + (marine.y - roach.y)**2) 
                    print("Distance: " + str(dist))
                    

    def stop(self):
        self.stop_signal = True
