"""Train YOLOv3 using tensorflow 2 on COCO dataset."""
import os

from absl import app, flags
from absl.flags import FLAGS
import tensorflow as tf
from tensorflow.keras import optimizers

from dataset import DataLoader, yolo_anchors, yolo_anchor_masks
from models import Yolov3, yolo_loss


flags.DEFINE_string('train_file', './data/processed/coco2017_train.tfrecord',
                    'Path to training dataset.')
flags.DEFINE_string('val_file', './data/processed/coco2017_val.tfrecord',
                    'Path to validation dataset.')
flags.DEFINE_integer('batch_size', 4, 'Batch size.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs.')
flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
flags.DEFINE_integer('image_size', 416, 'The image size.')
flags.DEFINE_integer('num_classes', 80, 'Number of classes.')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    # Prepare datasets
    train_tfrecord_file = os.path.abspath(FLAGS.train_file)
    train_dataset = DataLoader(batch_size=FLAGS.batch_size, training=True) \
        .from_files(train_tfrecord_file)

    val_tfrecord_file = os.path.abspath(FLAGS.val_file)
    val_dataset = DataLoader(batch_size=FLAGS.batch_size, training=False) \
        .from_files(val_tfrecord_file)

    # Build the model
    model = Yolov3(FLAGS.image_size, anchors=yolo_anchors,
                   masks=yolo_anchor_masks, training=True)
    model.summary()

    # Define our losses
    loss = [yolo_loss(yolo_anchors[mask], classes=FLAGS.num_classes)
            for mask in yolo_anchor_masks]
    optimizer = optimizers.Adam(lr=FLAGS.lr)

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss)

    # Training
    model.fit(train_dataset,
              batch_size=FLAGS.batch_size,
              # steps_per_epoch=num_train//FLAGS.batch_size,
              steps_per_epoch=1,
              epochs=FLAGS.epochs,
              validation_data=val_dataset,
              validation_steps=1)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
