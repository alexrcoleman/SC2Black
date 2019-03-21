from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.lib import actions
from pysc2.lib import features

import time

import numpy as np
import utils as U

_PLAYER_SELF = features.PlayerRelative.SELF

def run_loop(agent, env, max_frames=0):
  """A run loop to have agents and an environment interact."""
  start_time = time.time()

  try:
    while True:
      num_frames = 0
      obs = env.reset()[0]
      # phase = 1
      last_net_act_id = 0
      while True:
        num_frames += 1
        last_obs = obs
        norm_step = num_frames/max_frames
        last_act_onehot = np.zeros([len(U.useful_actions)], dtype=np.float32)
        last_act_onehot[last_net_act_id] = 1
        obs.observation.custom_inputs = np.concatenate([[norm_step],last_act_onehot], axis = 0)

        # action 2 (select_point) will select indanger marine if focus fire is on,
        # so this sets its availability
        if agent.force_focus_fire:
          dmarine = U.getDangerousMarineLoc(obs)
          if dmarine is None:
            index = np.argwhere(obs.observation.available_actions==2)
            obs.observation.available_actions = np.delete(obs.observation.available_actions, index)

        last_net_act_id, act = agent.step(obs)

        obs = env.step([act])[0]
        ### HACKY FIX !!!
        obs.observation.custom_inputs = np.concatenate([[norm_step],last_act_onehot], axis = 0)

        is_done = (num_frames >= max_frames) or obs.last()

        yield [last_obs, last_net_act_id, act, obs], is_done
        if is_done:
          break
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)
