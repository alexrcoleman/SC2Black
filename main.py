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
from pysc2.lib import stopwatch
import tensorflow as tf
from run_loop import run_loop
import datetime
import statusgui
from tkinter import Tk

COUNTER = 0
LOCK = threading.Lock()
FLAGS = flags.FLAGS
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", 2e-5, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("max_steps", int(1e5), "Total steps for training.")
flags.DEFINE_integer("snapshot_step", int(1e3), "Step for snapshot.")
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.DEFINE_string("device", "0", "Device for training.")

flags.DEFINE_string("map", "DefeatRoaches", "Name of a map to use.")
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "agents.a3c_agent.A3CAgent", "Which agent to run.")
flags.DEFINE_string("net", "fcn", "atari or fcn.")
flags.DEFINE_integer("max_agent_steps", 500, "Total agent steps.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")
flags.DEFINE_bool("force_focus_fire", False, "Whether to force focus firing (i.e. all 'attack' actions will redirect to lowest health / closest roach)")
flags.DEFINE_bool("use_tensorboard", False, "weather or not to use usetensorboard")
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
  DEVICE = ['/cpu:0']
else:
  PARALLEL = 1
  MAX_AGENT_STEPS = (120 * 16) // FLAGS.step_mul
  DEVICE = ['/cpu:0']

LOG = FLAGS.log_path+FLAGS.map+'/'+FLAGS.net
SNAPSHOT = FLAGS.snapshot_path+FLAGS.map+'/'+FLAGS.net

if not os.path.exists(LOG):
  os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
  os.makedirs(SNAPSHOT)

average = 0
def run_thread(agent, map_name, visualize, summary_writer):
  global COUNTER, average
  with sc2_env.SC2Env(
    map_name=map_name,
    agent_interface_format=sc2_env.parse_agent_interface_format(
      feature_screen=FLAGS.screen_resolution,
      feature_minimap=FLAGS.minimap_resolution,
      use_feature_units=True),
    step_mul=FLAGS.step_mul,
    visualize=visualize) as env:
    # env = available_actions_printer.AvailableActionsPrinter(env)
    # Only for a single player!
    replay_buffer = []
    update_steps = 20
    game_step = 0
    for recorder, is_done in run_loop(agent, env, MAX_AGENT_STEPS):
      game_step += 1
      if FLAGS.training:
        replay_buffer.append(recorder)
        if is_done:
          counter = 0
          with LOCK:
            COUNTER += 1
            counter = COUNTER
          # Learning rate schedule
          learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)
          agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)
          obs = recorder[-1].observation
          score = obs["score_cumulative"][0]
          print('Your score is '+str(score)+'! counter is ' + str(counter))
          if counter % 100 == 0:
            print('Average is ' + str(average))
            average = score
          else:
            average = (average * ((counter - 1) % 100) + score)/(counter % 100)

          replay_buffer = []
          if counter % FLAGS.snapshot_step == 1:
            agent.save_model(SNAPSHOT, counter)
          if counter >= FLAGS.max_steps:
            break
        elif update_steps % game_step == 0:
            counter = 0
            with LOCK:
              COUNTER += 1
              counter = COUNTER
            learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)
            agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)
            replay_buffer = []

      elif is_done:
        obs = recorder[-1].observation
        score = obs["score_cumulative"][0]
        print('Your score is '+str(score)+'!')
      if is_done:
         addScore(obs["score_cumulative"][0], summary_writer)


      if agent.killed:
        break

    if FLAGS.save_replay:
      env.save_replay(agent.name)

def addScore(score, summary_writer):
    global scoreQueue, COUNTER
    sum = tf.Summary()
    sum.value.add(tag='score',simple_value=score)
    with LOCK:
        summary_writer.add_summary(sum, COUNTER)
        scoreQueue.append(score)

def runningAverage(list, size):
    cnt = 0
    sum = 0
    nlist = []
    for i in range(len(list)):
        sum += list[i]
        if i >= size:
            sum -= list[i-size]
        if i+1 >= size:
            nlist.append(sum/size)
    return nlist


def printList(statList):
  # Open text file
  stats = open("scoreStatistic.csv", mode='w')
  for x in statList:
     stats.write(str(x) + ",")

def process_cmd(agents):
  while True:
    line = input()
    if line == 'q' or line == 'exit' or line == 'quit':
      print("Killing agents...")
      for a in agents:
        a.killed = True
      break
    else:
      print("Unknown command '" + line + "'")
def _main(unused_argv):
  """Run agents"""
  stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
  stopwatch.sw.trace = FLAGS.trace

  maps.get(FLAGS.map)  # Assert the map exists.

  # Setup agents
  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = getattr(importlib.import_module(agent_module), agent_name)

  agents = []
  summary_writer = tf.summary.FileWriter(TBDIR)
  for i in range(PARALLEL):
    # TODO: add flag to make 1/4 be testing
    agent = agent_cls(FLAGS.training and (i % 40 != 39), FLAGS.screen_resolution, FLAGS.force_focus_fire, summary_writer, FLAGS.use_tensorboard, 'A3CAgent' + str(i))
    agent.build_model(i > 0, DEVICE[i % len(DEVICE)])
    agents.append(agent)

  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  for i in range(PARALLEL):
    agents[i].setup(sess, summary_writer)

  agent.initialize()
  if not FLAGS.training or FLAGS.continuation:
    global COUNTER
    COUNTER = agent.load_model(SNAPSHOT)
  global scoreQueue
  scoreQueue = []

  # Run threads
  threads = []
  for i in range(PARALLEL):
    t = threading.Thread(target=run_thread, args=(agents[i], FLAGS.map, i == 0 and FLAGS.render, summary_writer))
    threads.append(t)
    t.daemon = True
    t.start()
    time.sleep(5)
  cmdThread = threading.Thread(target=process_cmd, args=(agents,))
  cmdThread.daemon = True
  cmdThread.start()

  # Setup GUI
  root = Tk()
  my_gui = statusgui.StatusGUI(root, agents)
  def updGUI():
    my_gui.update()
    root.after(50, updGUI)
  updGUI()
  root.mainloop()

  # Join threads once the user clicks exit or quit (should all be dying)
  for t in threads:
    t.join()
  summary_writer.close()
  printList(scoreQueue)

  steps = min(MAX_AGENT_STEPS * FLAGS.step_mul, 1920)
  print("Showing score graphs")
  plt.title("Score over Time - %s x%d %.1fs" % (FLAGS.map, FLAGS.parallel, steps / 16))
  plt.ylabel('Score')
  plt.xlabel('Time')
  plt.plot(scoreQueue)
  plt.plot(runningAverage(scoreQueue, 8))

  plt.legend(['Score', '8-game average'])
  plt.show()

  if FLAGS.profile:
    print(stopwatch.sw)


if __name__ == "__main__":
  app.run(_main)
