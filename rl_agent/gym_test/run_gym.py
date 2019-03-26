from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../../')
import gym
import numpy as np
from absl import flags
from gym_brain import Brain
from optimizer import Optimizer
from gym_environment import Environment
import datetime
import gym_statusgui
import tensorflow as tf
import time

FLAGS = flags.FLAGS
flags.DEFINE_string('environment','CartPole-v0','Which gym environment to run')
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_float("learning_rate", 7e-5, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_float("entropy_rate", .15, "entropy weight")
flags.DEFINE_bool("use_tensorboard", True,
                  "whether or not to use usetensorboard")
FLAGS(sys.argv)

def run_gym_test():
    summary_writer = createSummaryWriter()
    env = gym.make(FLAGS.environment)
    input_shape = env.observation_space.shape
    # (TODO) fix this to not be constant
    output_shape = (2,)

    env.close()
    brain = Brain(FLAGS, summary_writer,input_shape, output_shape)

    envs = [Environment(FLAGS, brain, summary_writer,
                    'A3CAgent_' + str(i)) for i in range(FLAGS.parallel)]
    opts = [Optimizer(brain) for i in range(2)]

    for o in opts:
        o.daemon = True
        o.start()
    for e in envs:
        e.daemon = True
        e.start()
        time.sleep(5)
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
    title = FLAGS.environment + "_" + stamp + "_x" + str(FLAGS.parallel)
    TBDIR = './gym_tb/' + title
    return tf.summary.FileWriter(TBDIR)

def createGUI(envs):
    root = Tk()
    my_gui = statusgui.StatusGUI(root, envs)

    def updGUI():
        my_gui.update()
        root.after(100, updGUI)
    updGUI()
    root.mainloop()

run_gym_test()
