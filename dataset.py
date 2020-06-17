"""Yolo v3 dataset loader."""
import numpy as np
import tensorflow as tf


yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


class DataLoader:
    def __init__(self, anchors=yolo_anchors,
                 anchor_masks=yolo_anchor_masks,
                 batch_size=32, image_size=416,
                 max_boxes=100, channels=3,
                 training=True):
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_boxes = max_boxes
        self.channels = channels
        self.training = training

    def from_files(self, tfrecord_files):
        """Create dataset from the given files."""
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        if self.training:
            dataset = dataset.shuffle(1000) \
                .map(self.parse_fn,
                     num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                .batch(self.batch_size) \
                .map(lambda x, y: (
                    transform_images(x),
                    transform_targets(y, self.anchors,
                                      self.anchor_masks,
                                      self.image_size)),
                     num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                .prefetch(tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(
                    self.parse_fn,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                .batch(self.batch_size) \
                .map(lambda x, y: (
                    transform_images(x),
                    transform_targets(y, self.anchors,
                                      self.anchor_masks,
                                      self.image_size)),
                     num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                .prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def parse_fn(self, example):
        example_fmt = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/text': tf.io.VarLenFeature(tf.string),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64)
        }
        parsed_example = tf.io.parse_single_example(example, example_fmt)
        image = tf.image.decode_jpeg(parsed_example['image/encoded'],
                                     channels=self.channels)
        image = tf.image.resize(image, (self.image_size, self.image_size))
        # height = parsed_example['image/height']
        # width = parsed_example['image/width']
        labels = tf.cast(
            tf.sparse.to_dense(parsed_example['image/object/class/label']),
            tf.float32)
        y_train = tf.stack(
            [tf.sparse.to_dense(parsed_example['image/object/bbox/xmin']),
             tf.sparse.to_dense(parsed_example['image/object/bbox/ymin']),
             tf.sparse.to_dense(parsed_example['image/object/bbox/xmax']),
             tf.sparse.to_dense(parsed_example['image/object/bbox/ymax']),
             labels], axis=1)
        # text = tf.sparse.to_dense(parsed_example['image/object/class/text'])
        paddings = [[0, self.max_boxes - tf.shape(y_train)[0]], [0, 0]]
        y_train = tf.pad(y_train, paddings)

        return image, y_train


def transform_images(image):
    image /= 255.
    return image


def transform_targets(y_train, anchors, anchor_masks, size):
    """Transform bounding boxes to yolo output.

    Args
    :y_train: A tensor of shape (N, max_boxes, 5) for the bounding
      boxes. The last dimension including (x1, y1, x2, y2, class).
    :anchors: A list of 9 2-tuples for the anchors.
    :anchor_masks: A list for anchor masks.
    :size: An integer for image size.

    Returns
        A tuple of 3 tensors. Each tensor which has shape of
        (N, grid, grid, anchors, 6) is a yolo output.
    """
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    # Create 3 yolo output
    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    """Transform the bounding boxes and corresponding best anchors
    to yolo output actually.

    Args
    :y_true: A tensor of shape (N, max_boxes, 6).
    :grid_size: The current grid size.
    :anchor_idxs: A tuple of 3 anchor mask indices.

    Returns
        A tuple of 3 tensors. Each tensor which has shape of
        (N, grid, grid, anchors, 6) is a yolo output.
    """
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())
