from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np

from absl import app
from absl import flags
from pysc2 import maps
import tensorflow as tf
import datetime
import statusgui
from tkinter import Tk
import utils as U
import time

from environment import Environment
from brain import Brain
from optimizer import Optimizer

FLAGS = flags.FLAGS
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", 7e-5, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("max_train_steps", int(1e6), "Total steps for training.")
flags.DEFINE_integer("snapshot_step", int(5e3), "Step for snapshot.")
flags.DEFINE_float("entropy_rate", .1, "entropy weight")
flags.DEFINE_string("map", "DefeatRoaches", "Name of a map to use.")
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("max_agent_steps", 500, "Total agent steps.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_bool("force_focus_fire", False,
                  "Whether to force focus firing (i.e. all 'attack' actions will redirect to lowest health / closest roach)")
flags.DEFINE_bool("use_tensorboard", True,
                  "whether or not to use usetensorboard")
flags.DEFINE_string("tensorboard_dir", None, "directory for tb")
FLAGS(sys.argv)

def _main(unused_argv):
    maps.get(FLAGS.map)  # Assert the map exists.
    SNAPSHOT = './snapshot/' + FLAGS.map
    if not os.path.exists(SNAPSHOT):
        os.makedirs(SNAPSHOT)

    summary_writer = createSummaryWriter()
    if FLAGS.training:
        max_agent_steps = min(FLAGS.max_agent_steps, (120 * 16) // FLAGS.step_mul)
    else:
        max_agent_steps = (120 * 16) // FLAGS.step_mul

    brain = Brain(FLAGS, summary_writer)
    if not FLAGS.training or FLAGS.continuation:
        brain.load_model(SNAPSHOT)

    envs = [Environment(FLAGS, brain, max_agent_steps, summary_writer,
                        'A3CAgent_' + str(i)) for i in range(FLAGS.parallel)]
    opts = [Optimizer(brain) for i in range(2)]

    for o in opts:
        o.daemon = True
        o.start()
    for e in envs:
        e.daemon = True
        e.start()
        time.sleep(6)
    print("All threads started")
    createGUI(envs)
    print("Stopping threads")
    for o in opts:
        o.stop()
    for e in envs:
        e.stop()

    for e in envs:
        e.join()
    summary_writer.close()


def createSummaryWriter():
    TBDIR = ''
    date = datetime.datetime.now()
    stamp = date.strftime('%Y.%m.%d_%H.%M')
    title = FLAGS.map + "_" + stamp + "_x" + str(FLAGS.parallel)
    if FLAGS.tensorboard_dir != None:
        title = title + "_" + FLAGS.tensorboard_dir
    TBDIR = './tboard/' + title
    return tf.summary.FileWriter(TBDIR)


def createGUI(envs):
    root = Tk()
    my_gui = statusgui.StatusGUI(root, envs)

    def updGUI():
        my_gui.update()
        root.after(100, updGUI)
    updGUI()
    root.mainloop()
    print("Done main loop")


if __name__ == "__main__":
    app.run(_main)
