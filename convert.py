"""
"""
import logging
from absl import app, flags
from absl.flags import FLAGS

import numpy as np
import tensorflow as tf
from models import Yolov3
from utils import load_darknet_weights


flags.DEFINE_string('weights', './yolov3.weights', 'Yolov3 weights file')
flags.DEFINE_string('output', './checkpoints/yolov3.tf',
                    'Checkpoint output file.')
flags.DEFINE_integer('num_classes', 80, 'Number of classes')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    yolo = Yolov3(classes=FLAGS.num_classes)
    yolo.summary()
    logging.info('model created')

    load_darknet_weights(yolo, FLAGS.weights)
    logging.info('weights loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    yolo(img)
    logging.info('sanity check passed')

    yolo.save_weights(FLAGS.output)
    logging.info('weights saved')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
