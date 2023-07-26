import os
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
import cv2
import argparse
from tqdm import tqdm
import random

def yoloize_boxes(box):
    """
    box = [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = abs(x1 - x2)
    height = abs(y1 - y2)

    return [x_center, y_center, width, height]

def get_distances(location):
    return np.sqrt(np.sum(np.array(location) ** 2))

def partition_data(filenames):
    partition_index = int(len(filenames) * 0.8)
    random.Random(10).shuffle(filenames)
    return filenames[:partition_index], filenames[partition_index:]

def convert_and_save_labels(input_files, include_distance, save_dir, 
                            extension=None):
    # Update path for this set
    label_dir = os.path.join(save_dir, 'labels')
    if extension:
        label_dir = os.path.join(label_dir, extension)
    os.makedirs(label_dir, exist_ok=True)

    for filename in tqdm(input_files):
        yolo_labels = defaultdict(list)
        local_labels = pd.read_csv(
            filename, header=None, delim_whitespace=True)
        for _, row in local_labels.iterrows():
            if row[0] == 'DontCare':
                continue
            yolo_labels['type'].append(numerical_mapping[row[0]])
            boxes = yoloize_boxes(list(row[4:8]))
            yolo_labels['x_center'].append(boxes[0])
            yolo_labels['y_center'].append(boxes[1])
            yolo_labels['width'].append(boxes[2])
            yolo_labels['height'].append(boxes[3])
            if include_distance:
                yolo_labels['distance'].append(get_distances(list(row[11:14])))
        yolo_df = pd.DataFrame(yolo_labels)
        # append the labels extension 
        basename = os.path.basename(filename)
        save_path = os.path.abspath(os.path.join(label_dir, basename))
        yolo_df.to_csv(save_path, sep=' ', index=False, header=False)

if __name__=='__main__':
    columns = ['type', # str
               'truncated', # float
               'occluced', # int
               'alpha', # float 
               'bbox', # list(float(4))
               'dimensions', # list(float(3)) 
               'location', # list(float(3)) 
               'rotation'] # float
    
    numerical_mapping = {'Car': 0, 'Pedestrian': 1, 'Misc': 2, 'Tram': 3,
        'Van': 4, 'Truck': 5, 'Cyclist': 6, 'Person_sitting': 7}
        
    parser = argparse.ArgumentParser()

    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, required=True, 
        help="Select location for saving data. New directories 'labels'"
        "and 'imagery' (assuming an image dir was given) will be created and"
        "'train' and 'valid' directories will be created within those")
    parser.add_argument('--include_distance', action='store_true')

    args = parser.parse_args()

    filenames = [os.path.join(args.label_dir, f) 
        for f in os.listdir(args.label_dir) if f.endswith('.txt')]
    
    train_files, val_files = partition_data(filenames)

    # Save train files
    print("Converting training data")
    convert_and_save_labels(
        train_files, args.include_distance, args.save_dir, extension='train')

    # Save val files
    print("Converting validation data")
    convert_and_save_labels(
        val_files, args.include_distance, args.save_dir, extension='valid')
    
    if args.image_dir:
        imagenames = os.listdir(args.image_dir)
        
        train_files = [os.path.basename(f).split('.')[0] for f in train_files]
        val_files = [os.path.basename(f).split('.')[0] for f in val_files]

        train_images = [os.path.join(args.image_dir, img) 
            for img in imagenames if img.split('.')[0] in train_files]
        val_images = [os.path.join(args.image_dir, img) 
            for img in imagenames if img.split('.')[0] in val_files]

        save_location = os.path.join(args.save_dir, 'images')        
        train_location = os.path.join(save_location, 'train')
        os.makedirs(train_location, exist_ok=True)
        print("Moving training imagery")
        for train_image in tqdm(train_images):
            image = cv2.imread(train_image)
            image_name = os.path.basename(train_image)
            cv2.imwrite(os.path.join(train_location, image_name), image)

        val_location = os.path.join(save_location, 'valid')
        os.makedirs(val_location, exist_ok=True)
        print("Moving validation imagery")
        for val_image in tqdm(val_images):
            image = cv2.imread(val_image)
            image_name = os.path.basename(val_image)
            cv2.imwrite(os.path.join(val_location, image_name), image)