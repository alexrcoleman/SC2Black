from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import importlib
import threading
import matplotlib.pyplot as plt
import numpy as np

from absl import app
from absl import flags
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
import tensorflow as tf
import datetime
import statusgui
from tkinter import Tk
import utils as U
from Environment import Environment

FLAGS = flags.FLAGS
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", 2e-5, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("max_steps", int(1e6), "Total steps for training.")
flags.DEFINE_integer("snapshot_step", int(1e3), "Step for snapshot.")
flags.DEFINE_float("entropy_rate", .03, "entropy weight")
flags.DEFINE_string("map", "DefeatRoaches", "Name of a map to use.")
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("max_agent_steps", 500, "Total agent steps.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")
flags.DEFINE_bool("force_focus_fire", False, "Whether to force focus firing (i.e. all 'attack' actions will redirect to lowest health / closest roach)")
flags.DEFINE_bool("use_tensorboard", True, "weather or not to use usetensorboard")
flags.DEFINE_string("tensorboard_dir", None, "directory for tb")
FLAGS(sys.argv)

TBDIR = ''
if FLAGS.use_tensorboard:
    date = datetime.datetime.now()
    stamp = date.strftime('%Y.%m.%d_%H.%M')
    title = FLAGS.map + "_" + stamp + "_x" + str(FLAGS.parallel)
    if FLAGS.tensorboard_dir != None:
      title = title + "_" + FLAGS.tensorboard_dir
    TBDIR = './tboard/' + title


if FLAGS.training:
  PARALLEL = FLAGS.parallel
  MAX_AGENT_STEPS = min(FLAGS.max_agent_steps, (120 * 16) // FLAGS.step_mul)
else:
  PARALLEL = 1
  MAX_AGENT_STEPS = (120 * 16) // FLAGS.step_mul


SNAPSHOT = './snapshot/'+FLAGS.map
if not os.path.exists(SNAPSHOT):
  os.makedirs(SNAPSHOT)

def process_cmd(envs):
  while True:
    line = input()
    if line == 'q' or line == 'exit' or line == 'quit':
      print("Killing agents...")
      for env in envs:
        env.stop_signal = True
      break
    else:
      print("Unknown command '" + line + "'")

def _main(unused_argv):
  """Run agents"""

  maps.get(FLAGS.map)  # Assert the map exists.

  # Open text file
  stats = open("scoreStatistic.txt", mode='w')

  envs = []
  summary_writer = tf.summary.FileWriter(TBDIR)
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  for i in range(PARALLEL):
    env = Environment(FLAGS, i > 0, MAX_AGENT_STEPS, summary_writer, 'A3CAgent_' + str(i), sess)
    envs.append(env)

  envs[0].agent.initialize()
  if not FLAGS.training or FLAGS.continuation:
    Environment.counter = envs[0].agent.load_model(SNAPSHOT)
  for env in envs:
    env.start()
    time.sleep(5)
  cmdThread = threading.Thread(target=process_cmd, args=(envs,))
  cmdThread.daemon = True
  cmdThread.start()

  # Setup GUI
  root = Tk()
  my_gui = statusgui.StatusGUI(root, envs)
  def updGUI():
    my_gui.update()
    root.after(50, updGUI)
  updGUI()
  root.mainloop()

  # Join threads once the user clicks exit or quit (should all be dying)
  for t in envs:
    t.join()
  summary_writer.close()

  steps = min(MAX_AGENT_STEPS * FLAGS.step_mul, 1920)



if __name__ == "__main__":
  app.run(_main)
