import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

curr_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.abspath(curr_dir + '/../data/processed')
tfrecord_file = os.path.join(dataset_dir, 'coco2017_val.tfrecord')


max_boxes = 50
image_size = 416
channels = 3
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


def parse_fn(example):
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
                                 channels=channels)
    image = tf.image.resize(image, (image_size, image_size))
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
    paddings = [[0, max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)
    return image, y_train


def transform_images(image):
    image /= 255.
    return image


def transform_targets(y_train, anchors, anchor_masks, size):
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

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
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


dataset = tf.data.TFRecordDataset(tfrecord_file)
dataset = dataset.map(parse_fn).shuffle(10).batch(5) \
        .map(lambda x, y: (x, transform_targets(y, yolo_anchors,
                                               yolo_anchor_masks, image_size)))

for batch in dataset.take(1):
    pass


images = batch[0].numpy()
y_train = batch[1]
width, height = 416, 416

image = images[0].astype('uint8')
image = Image.fromarray(image)
draw = ImageDraw.Draw(image)
for x1, y1, x2, y2, class_id in y_train[0]:
    x1, y1 = int(x1 * width), int(y1 * height)
    x2, y2 = int(x2 * width), int(y2 * height)
    draw.rectangle(((x1, y1), (x2, y2)), outline=(0, 0, 255))
    draw.text((x1, y1-10), str(class_id), (0, 0, 255, 255))
del draw
plt.imshow(image)
plt.show()
