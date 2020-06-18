"""Test Yolov3 model on real image."""
from absl import app, flags
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf

from models import Yolov3
from dataset import transform_images
from utils import draw_bboxes


flags.DEFINE_string('model_path', './checkpoints/yolov3.tf',
                    'Path to model checkpoints.')
flags.DEFINE_string('image_path', '', 'Path to input image.')
flags.DEFINE_integer('image_size', 416, 'The image size.')
flags.DEFINE_integer('num_classes', 80, 'Number of classes.')
flags.DEFINE_string('class_file', './data/coco.names', 'Class name file.')
flags.DEFINE_boolean('show', False, 'Show image before saving')
flags.DEFINE_string('output', 'predictions.jpg', 'Output file')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    # Load coco names
    coco_map = {i: name.strip() for i, name in enumerate(open(FLAGS.class_file))}

    # Load the image and preprocess
    image_size = FLAGS.image_size
    image_raw = tf.image.decode_jpeg(
        open(FLAGS.image_path, 'rb').read(), channels=3)
    image = tf.cast(image_raw, tf.float32)
    image = transform_images(image)
    image = tf.image.resize(image, (image_size, image_size))
    images = tf.expand_dims(image, axis=0)

    # Load the model
    model = Yolov3(classes=FLAGS.num_classes)
    model.load_weights(FLAGS.model_path).expect_partial()

    # Predict in eager mode
    boxes, scores, classes, nums = model(images)

    nums = int(nums[0])
    boxes = np.array(boxes[0])[:nums]
    scores = np.array(scores[0])[:nums]
    classes = [coco_map[idx] for idx in np.array(classes[0])[:nums]]

    image = cv2.cvtColor(image_raw.numpy(), cv2.COLOR_RGB2BGR)
    image = draw_bboxes(image, boxes, scores, classes)
    if FLAGS.show:
        cv2.imshow('img', image)
        cv2.waitKey(0)
    cv2.imwrite(FLAGS.output, image)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
