"""Get output feature map shape
Make sure add forward graph only
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import glob

import numpy as np
import tensorflow as tf

from configs import *
from nets import *


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint', '/home/zehao/PycharmProjects/squeezeDet/data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")


def get_feamap_shape():
    """Get output feature map shape."""

    with tf.Graph().as_default():
        # Load model
        mc = voc_squeezeDet_5D_allmaxpool_config()
        mc.BATCH_SIZE = 1
        # model parameters will be restored from checkpoint
        mc.LOAD_PRETRAINED_MODEL = False
        model = VocSqueezeDet5DAMP(mc, 0)

        #saver = tf.train.Saver(model.model_params)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            im = np.zeros([100, 100, 3])
            im = im.astype(np.float32, copy=False)
            im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))

            # Detect
            feamap = sess.run(model.preds, feed_dict={model.image_input: [im], model.keep_prob: 1.0})
            print("shape: {}".format(np.shape(feamap)))


def main(argv=None):
    get_feamap_shape()


if __name__ == '__main__':
    tf.app.run()
