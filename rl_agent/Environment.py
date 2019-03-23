from agents.a3c_agent import A3CAgent
from pysc2.env import sc2_env
import numpy as np
import threading
import utils as U
import tensorflow as tf

class Environment(threading.Thread):
    counter = 0
    average = 0
    UPDATE_STEPS = 30
    LOCK = threading.Lock()

    def __init__(self, flags, reuse, max_frames, summary_writer, name, sess):
        threading.Thread.__init__(self)
        self.summary_writer = summary_writer
        self.flags = flags
        self.max_frames = max_frames
        self.stop_signal = False
        # TODO: Fix this
        self.agent = A3CAgent(flags, 64, summary_writer, sess, name)
        self.agent.build_model(reuse, '/gpu:0')


    def run(self):
        # Sets up interaction with environment
        agent_interface_format = sc2_env.parse_agent_interface_format(
        feature_screen=64,
        feature_minimap=64,
        use_feature_units=True)

        # Sets up environment
        with sc2_env.SC2Env(map_name=self.flags.map,
                agent_interface_format=agent_interface_format,
                step_mul=self.flags.step_mul,
                visualize=self.flags.render) as self.env:
            while not self.stop_signal:
                self.run_game()

    def run_game(self):
        FLAGS = self.flags
        num_frames = 0
        replay_buffer = []
        timestep = self.env.reset()[0]
        last_net_act_id = 0

        while not self.stop_signal:
          num_frames += 1
          last_timestep = timestep
          norm_step = num_frames/self.max_frames
          last_act_onehot = np.zeros([len(U.useful_actions)], dtype=np.float32)
          last_act_onehot[last_net_act_id] = 1
          timestep.observation.custom_inputs = np.concatenate([[norm_step],last_act_onehot], axis = 0)

          # action 2 (select_point) will select indanger marine if focus fire is on,
          # so this sets its availability
          if FLAGS.force_focus_fire:
            dmarine = U.getDangerousMarineLoc(timestep)
            if dmarine is None:
              index = np.argwhere(timestep.observation.available_actions==2)
              timestep.observation.available_actions = np.delete(timestep.observation.available_actions, index)

          last_net_act_id, act = self.agent.act(timestep)

          timestep = self.env.step([act])[0]
          ### HACKY FIX !!!
          timestep.observation.custom_inputs = np.concatenate([[norm_step],last_act_onehot], axis = 0)

          is_done = (num_frames >= self.max_frames) or timestep.last()

          local_counter = 0
          if FLAGS.training:
              replay_buffer.append([last_timestep, last_net_act_id, act, timestep])
              if is_done or num_frames % Environment.UPDATE_STEPS == 0:
                  with Environment.LOCK:
                      Environment.counter += 1
                      local_counter = Environment.counter
                  learning_rate = FLAGS.learning_rate * (1 - 0.9 * local_counter / FLAGS.max_steps)
                  entropy_rate = FLAGS.entropy_rate * (1 - 0.99 * local_counter / FLAGS.max_steps)
                  self.agent.update(replay_buffer, FLAGS.discount, learning_rate, local_counter, entropy_rate)
                  replay_buffer = []
                  if local_counter % FLAGS.snapshot_step == 1:
                      self.agent.save_model('./snapshot/'+FLAGS.map, local_counter)
                  if local_counter >= FLAGS.max_steps:
                      break
          if is_done:
              score = timestep.observation["score_cumulative"][0]
              sum = tf.Summary()
              sum.value.add(tag='score',simple_value=score)
              with Environment.LOCK:
                  self.summary_writer.add_summary(sum, local_counter)
              print('counter '+str(local_counter)+' score: ' + str(score))
              break
