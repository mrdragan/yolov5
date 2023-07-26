import os
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

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
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--include_distance', action='store_true')

    args = parser.parse_args()

    for filename in tqdm(os.listdir(args.label_dir)):
        yolo_labels = defaultdict(list)
        full_path = os.path.abspath(os.path.join(args.label_dir, filename))
        local_labels = pd.read_csv(
            full_path, header=None, delim_whitespace=True)
        for index, row in local_labels.iterrows():
            if row[0] == 'DontCare':
                continue
            yolo_labels['type'].append(numerical_mapping[row[0]])
            boxes = yoloize_boxes(list(row[4:8]))
            yolo_labels['x_center'].append(boxes[0])
            yolo_labels['y_center'].append(boxes[1])
            yolo_labels['width'].append(boxes[2])
            yolo_labels['height'].append(boxes[3])
            if args.include_distance:
                yolo_labels['distance'].append(get_distances(list(row[11:14])))
        yolo_df = pd.DataFrame(yolo_labels)
        save_path = os.path.abspath(os.path.join(args.save_dir, filename))
        yolo_df.to_csv(save_path, sep=' ', index=False, header=False)


