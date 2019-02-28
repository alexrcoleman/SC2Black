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
      last_actions = 0
      last_net_act_id = 0
      while True:
        num_frames += 1
        last_obs = obs
        norm_step = num_frames/max_frames
        last_act_onehot = np.zeros([len(U.useful_actions)], dtype=np.float32)
        last_act_onehot[last_net_act_id] = 1
        obs.observation.custom_inputs = np.concatenate([[norm_step],last_act_onehot], axis = 0)
        # if phase == 0:
            # some code to alternate between units, needs some work though
            # marines = [unit for unit in timesteps[0].observation.feature_units if unit.alliance == _PLAYER_SELF]
            # marine_unit = next(m for m in marines if not m.is_selected)
            # marine_xy = [marine_unit.x, marine_unit.y]
            # acts = [actions.FUNCTIONS.select_point("select", marine_xy)]
        # else:
        act = agent.step(obs)

        last_act_id = act.function
        last_net_act_id = U.compress_actions[last_act_id]

        # phase = 1 - phase
        obs = env.step([act])[0]

        # Real time (technically should be 1/16 * steps_mul)
        # if False:
        #   time.sleep(.25)

        is_done = (num_frames >= max_frames) or obs.last()

        yield [last_obs, act, obs], is_done
        if is_done:
          break
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)
