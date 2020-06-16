import os
import tensorflow as tf
from context import Yolov3
from context import yolo_anchors, yolo_anchor_masks, DataLoader


physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)


image_size = 416
tfrecord_file = os.path.abspath('data/processed/coco2017_train.tfrecord')
dataset = DataLoader(yolo_anchors, yolo_anchor_masks,
                  batch_size=4, training=True).from_files(tfrecord_file)
model = Yolov3(image_size, channels=3, anchors=yolo_anchors,
               masks=yolo_anchor_masks, training=False)

for batch in dataset.take(1):
    pass
pred = model(batch[0])
print(pred)
