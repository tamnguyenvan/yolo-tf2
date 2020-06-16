import os
import glob
import json
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


def visualize(image, bboxes, class_names):
    draw = ImageDraw.Draw(image)
    for (x, y, w, h), class_name in zip(bboxes, class_names):
        draw.rectangle(((x, y), (x + w, y + h)), outline=(0, 0, 255))
        draw.text((x, y-10), class_name, fill=(255, 255, 255, 255))
    del draw
    image = np.array(image, dtype='uint8')
    plt.imshow(image)
    plt.show()


curr_dir = os.path.dirname(os.path.abspath(__file__))
annotations_dir = os.path.abspath(curr_dir + '/../data/raw/annotations/')
annotations_file = os.path.join(annotations_dir, 'instances_train2017.json')

annotations_json = json.load(open(annotations_file))
images = annotations_json['images']
annotations = annotations_json['annotations']
categories = annotations_json['categories']
category_meta = {}
for category_data in categories:
    class_id = category_data['id']
    class_name = category_data['name']
    category_meta[class_id] = class_name

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

image2annot_map = {}
for annotation_data in annotations:
    image_id = annotation_data['image_id']
    filename = image_meta[image_id]['filename']
    bbox = list(map(int, annotation_data['bbox']))
    class_id = annotation_data['category_id']
    class_name = category_meta[class_id]
    if image_id not in image2annot_map:
        image2annot_map[image_id] = {
            'filename': filename,
            'bbox': [bbox],
            'classes': [class_id],
            'names': [class_name]
        }
    image2annot_map[image_id]['bbox'].append(bbox)
    image2annot_map[image_id]['classes'].append(class_id)
    image2annot_map[image_id]['names'].append(class_name)

train_dir = os.path.abspath(curr_dir + '/../data/raw/train2017')

example_image_ids = list(image2annot_map.keys())[:5]
for example_image_id in example_image_ids:
    filename = image2annot_map[example_image_id]['filename']
    bboxes = image2annot_map[example_image_id]['bbox']
    classes = image2annot_map[example_image_id]['classes']
    class_names = image2annot_map[example_image_id]['names']

    image_path = os.path.join(train_dir, filename)
    image = Image.open(image_path)
    visualize(image, bboxes, class_names)
