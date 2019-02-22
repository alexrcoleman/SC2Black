from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.lib import actions
from pysc2.lib import features

import time


_PLAYER_SELF = features.PlayerRelative.SELF

def run_loop(agents, env, max_frames=0):
  """A run loop to have agents and an environment interact."""
  start_time = time.time()

  try:
    while True:
      num_frames = 0
      timesteps = env.reset()
      phase = 1
      last_actions = 0
      while True:
        num_frames += 1
        last_timesteps = timesteps
        acts = []
        timesteps[0].observation.phase = phase
        if phase == 0:
            # some code to alternate between units, needs some work though
            marines = [unit for unit in timesteps[0].observation.feature_units if unit.alliance == _PLAYER_SELF]
            marine_unit = next(m for m in marines if not m.is_selected)
            marine_xy = [marine_unit.x, marine_unit.y]
            acts = [actions.FUNCTIONS.select_point("select", marine_xy)]
        else:
            acts = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
            last_actions = acts
        # phase = 1 - phase
        timesteps = env.step(acts)

        # Real time (technically should be 1/16 * steps_mul)
        if False:
          time.sleep(.25)

        is_done = (num_frames >= max_frames) or timesteps[0].last()
        if last_actions != 0:
            yield [last_timesteps[0], last_actions[0], timesteps[0]], is_done
        if is_done:
          break
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)
