"""Create tfrecord files for training."""
import os
import json
import logging
from argparse import ArgumentParser
from sys import argv

import tensorflow as tf


logger = logging.getLogger(__name__)


def parse_args(argv):
    parser = ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val'], default='train',
                        help='Specify train or val split')
    parser.add_argument('--data_dir', type=str, default='../data/raw',
                        help='Data directory.')
    parser.add_argument('--output_file', type=str,
                        default='coco2017_train.tfrecord',
                        help='Path to output tfrecord')
    return parser.parse_args(argv)


def bytes_feature(value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def build_example(args, image_dir, image_id, image_data):
    """Build an example from annotation data."""
    filename = image_data['filename']
    img_path = os.path.join(image_dir, filename)
    img_raw = open(img_path, 'rb').read()

    width = image_data['width']
    height = image_data['height']
    bboxes = image_data['bboxes']
    class_ids = image_data['classes']
    class_names = image_data['names']

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    for bbox, class_id, class_name in zip(bboxes, class_ids, class_names):
        x, y, w, h = bbox
        xmin.append(float(x) / width)
        ymin.append(float(y) / height)
        xmax.append(float(x + w) / width)
        ymax.append(float(y + h) / height)
        classes_text.append(class_name.encode('utf8'))
        classes.append(class_id)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(value=height),
        'image/width': int64_feature(value=width),
        'image/filename': bytes_feature(value=filename.encode('utf-8')),
        'image/encoded': bytes_feature(value=img_raw),
        'image/object/bbox/xmin': float_feature(value=xmin),
        'image/object/bbox/ymin': float_feature(value=ymin),
        'image/object/bbox/xmax': float_feature(value=xmax),
        'image/object/bbox/ymax': float_feature(value=ymax),
        'image/object/class/text': bytes_feature(value=classes_text),
        'image/object/class/label': int64_feature(value=classes)
    }))
    return example


def main(args):
    # Load annotations
    dataset_dir = os.path.abspath(args.data_dir)
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    if args.split == 'train':
        annotations_file = os.path.join(
            annotations_dir, 'instances_train2017.json')
    else:
        annotations_file = os.path.join(
            annotations_dir, 'instances_val2017.json')
    logger.info('Loading {}'.format(annotations_file))

    annotations_json = json.load(open(annotations_file))
    images = annotations_json['images']
    annotations = annotations_json['annotations']
    categories = annotations_json['categories']

    # Create class_id -> class_name map
    category_meta = {}
    for category_data in categories:
        class_id = category_data['id']
        class_name = category_data['name']
        category_meta[class_id] = class_name

    # Create image_id -> (filename, width, height) map
    image_meta = {}
    for image_data in images:
        image_id = image_data['id']
        filename = image_data['file_name']
        width = image_data['width']
        height = image_data['height']
        image_meta[image_id] = {
            'filename': filename,
            'width': width,
            'height': height
        }

    # Create image_id -> annotation data map
    image2annot_map = {}
    for annotation_data in annotations:
        image_id = annotation_data['image_id']
        filename = image_meta[image_id]['filename']
        width = image_meta[image_id]['width']
        height = image_meta[image_id]['height']
        bbox = list(map(int, annotation_data['bbox']))
        class_id = annotation_data['category_id']
        class_name = category_meta[class_id]
        if image_id not in image2annot_map:
            image2annot_map[image_id] = {
                'filename': filename,
                'bboxes': [],
                'classes': [],
                'names': [],
                'width': width,
                'height': height
            }
        image2annot_map[image_id]['bboxes'].append(bbox)
        image2annot_map[image_id]['classes'].append(class_id)
        image2annot_map[image_id]['names'].append(class_name)

    # Create tfrecord file
    if args.split == 'train':
        image_dir = os.path.join(dataset_dir, 'train2017')
    else:
        image_dir = os.path.join(dataset_dir, 'val2017')

    logger.info('Loading images from {}'.format(image_dir))
    output_path = os.path.join(os.path.dirname(dataset_dir),
                               'processed', args.output_file)
    writer = tf.io.TFRecordWriter(output_path)
    for image_id, image_data in image2annot_map.items():
        tf_example = build_example(args, image_dir, image_id, image_data)
        writer.write(tf_example.SerializeToString())
    writer.close()
    logger.info("Done")


if __name__ == '__main__':
    main(parse_args(argv[1:]))
