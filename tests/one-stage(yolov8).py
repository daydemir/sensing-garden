# pip install ultralytics 
# python3 tests/one-stage(yolov8).py --weights /path/to/weights --data /path/to/dataset --conf 0.3 --iou 0.5

import argparse
import time
from ultralytics import YOLO
import yaml
import os

class YOLOTester:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def validate(self, data_path, conf, iou, num_frames):
        start_time = time.time()

        results = self.model.val(data=data_path, conf=conf, iou=iou)

        end_time = time.time()
        total_time = end_time - start_time

        avg_time_per_frame = total_time / num_frames 

        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per frame: {avg_time_per_frame:.4f} seconds")

        return results

def create_yaml_config(file_path, dataset_path, target_species):

    config = {
        'path': dataset_path,
        'train': '',
        'val': 'images',  # Validation images directory
        'test': '',
        'names': {i: name for i, name in enumerate(target_species)}
    }

    with open(file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def main():
    #input -> directory with images and labels subdirectories
    parser = argparse.ArgumentParser(description='YOLOv8 Validation Script')
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLO weights file')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory containing images and labels')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')

    args = parser.parse_args()

    num_frames = len(os.listdir(os.path.join(args.data, 'images')))

    target_species = ['insect']
    yaml_path = os.path.join(args.data, 'dataset.yaml')
    create_yaml_config(yaml_path, args.data, target_species)

    tester = YOLOTester(args.weights)
    tester.validate(yaml_path, args.conf, args.iou, num_frames)

if __name__ == '__main__':
    main()
