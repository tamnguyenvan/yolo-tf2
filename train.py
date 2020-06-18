"""Train YOLOv3 using tensorflow 2 on COCO dataset."""
import os
import logging

from absl import app, flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, TensorBoard,
    ModelCheckpoint, EarlyStopping
)

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
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager'], 'Training mode')


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

    if FLAGS.mode == 'eager':
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(FLAGS.epochs):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss)

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))
                logging.info('Epoch #{:04d} batch #{:05d}'
                             ' total loss {:.4f} {}'.format(
                                 epoch+1, batch, total_loss.numpy(),
                                 list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)

            avg_loss.reset_states()
    else:
        # Compile the model
        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint('./checkpoints/yolov3_train_{epoch}.tf',
                            verbose=1, save_weights_only=True),
            TensorBoard(log_dir='./logs')
        ]
        model.compile(optimizer=optimizer, loss=loss)

        # Training
        num_train = 117266
        num_val = 4952
        model.fit(train_dataset,
                  batch_size=FLAGS.batch_size,
                  steps_per_epoch=num_train//FLAGS.batch_size,
                  epochs=FLAGS.epochs,
                  validation_data=val_dataset,
                  validation_steps=num_val//FLAGS.batch_size,
                  callbacks=callbacks)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
