from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')
import gym
import numpy as np
from absl import flags
from gym_brain import Brain
from optimizer import Optimizer

FLAGS = flags.FLAGS
flags.DEFINE_string('environment','CartPole-v0','Which gym environment to run')

def run_gym_test():
    summary_writer = createSummaryWriter()
    brain = Brain(FLAGS, summary_writer, )

def createSummaryWriter():
    TBDIR = ''
    date = datetime.datetime.now()
    stamp = date.strftime('%Y.%m.%d_%H.%M')
    title = FLAGS.environment + "_" + stamp + "_x" + str(FLAGS.parallel)
    TBDIR = './gym_tb/' + title
    return tf.summary.FileWriter(TBDIR)
