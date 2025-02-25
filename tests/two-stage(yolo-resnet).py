# pip install ultralytics torchvision pillow numpy scikit-learn tabulate tqdm
# python3 tests/two-stage(yolo-resnet).py --data ' --yolo_weights --resnet_weights --use_resnet50

import os
import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.models import resnet152, resnet50
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
import time
import argparse
from collections import defaultdict
from tabulate import tabulate
from tqdm import tqdm
import csv

class TestTwoStage:
    def __init__(self, yolo_model_path, resnet_model_path, num_classes=22, use_resnet50=False, species_names=""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.yolo_model = YOLO(yolo_model_path)
        self.classification_model = resnet50(pretrained=False) if use_resnet50 else resnet152(pretrained=False)
        self.classification_model.fc = nn.Linear(self.classification_model.fc.in_features, num_classes)
        
        state_dict = torch.load(resnet_model_path, map_location=self.device)
        self.classification_model.load_state_dict(state_dict)
        self.classification_model.to(self.device)
        self.classification_model.eval()

        self.classification_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.species_names = species_names

    def get_frames(self, test_dir):
        image_dir = os.path.join(test_dir, "images")
        label_dir = os.path.join(test_dir, "labels")
        
        predicted_frames = []
        true_frames = []
        image_names = []

        start_time = time.time()  # Start timing

        for image_name in tqdm(os.listdir(image_dir), desc="Processing Images", unit="image"):
            image_names.append(image_name)
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt'))

            frame = cv2.imread(image_path)
            # Suppress print statements from YOLO model
            with torch.no_grad():
                results = self.yolo_model(frame, conf=0.3, iou=0.5, verbose=False)

            detections = results[0].boxes
            predicted_frame = []

            if detections:
                for box in detections:
                    xyxy = box.xyxy.cpu().numpy().flatten()
                    x1, y1, x2, y2 = xyxy[:4]
                    width = x2 - x1
                    height = y2 - y1
                    x_center = x1 + width / 2
                    y_center = y1 + height / 2

                    insect_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    insect_crop_rgb = cv2.cvtColor(insect_crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(insect_crop_rgb)
                    input_tensor = self.classification_transform(pil_img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        outputs = self.classification_model(input_tensor)

                    predicted_class_idx = outputs.argmax(dim=1).item()
                    img_height, img_width, _ = frame.shape
                    x_center_norm = x_center / img_width
                    y_center_norm = y_center / img_height
                    width_norm = width / img_width
                    height_norm = height / img_height
                    predicted_frame.append([predicted_class_idx, x_center_norm, y_center_norm, width_norm, height_norm])

            predicted_frames.append(predicted_frame if predicted_frame else [])

            true_frame = []
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                with open(label_path, 'r') as f:
                    for line in f:
                        label_line = line.strip().split()
                        true_frame.append([int(label_line[0]), *map(np.float32, label_line[1:])])

            true_frames.append(true_frame if true_frame else [])

        end_time = time.time()  # End timing

        with open("output.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Name", "True", "Predicted"])
            for image_name, true_frame, predicted_frame in zip(image_names, true_frames, predicted_frames):
                writer.writerow([image_name, true_frame, predicted_frame])
        
        return predicted_frames, true_frames, end_time - start_time
    
    def get_metrics(self, predicted_frames, true_frames):

        def calculate_iou(box1, box2):
            x1_min, y1_min = box1[1] - box1[3] / 2, box1[2] - box1[4] / 2
            x1_max, y1_max = box1[1] + box1[3] / 2, box1[2] + box1[4] / 2
            x2_min, y2_min = box2[1] - box2[3] / 2, box2[2] - box2[4] / 2
            x2_max, y2_max = box2[1] + box2[3] / 2, box2[2] + box2[4] / 2

            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)

            iou = inter_area / (box1_area + box2_area - inter_area)
            return iou

        def calculate_precision_recall(pred_boxes, true_boxes, iou_threshold=0.5):
            tp = 0
            fp = 0
            fn = 0

            matched_true_boxes = set()
            for pred_box in pred_boxes:
                matched = False
                for i, true_box in enumerate(true_boxes):
                    if i in matched_true_boxes:
                        continue
                    iou = calculate_iou(pred_box, true_box)
                    if iou >= iou_threshold and pred_box[0] == true_box[0]:
                        tp += 1
                        matched_true_boxes.add(i)
                        matched = True
                        break
                if not matched:
                    fp += 1

            fn = len(true_boxes) - len(matched_true_boxes)
            return tp, fp, fn

        species_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        background_metrics = {'tp': 0, 'fp': 0, 'fn': 0}

        for pred_frame, true_frame in zip(predicted_frames, true_frames):
            if not pred_frame and not true_frame:
                background_metrics['tp'] += 1
            elif not pred_frame:
                background_metrics['fn'] += 1
            elif not true_frame:
                background_metrics['fp'] += 1
            else:
                for pred_box in pred_frame:
                    species_idx = pred_box[0]
                    tp, fp, fn = calculate_precision_recall([pred_box], true_frame)
                    species_metrics[species_idx]['tp'] += tp
                    species_metrics[species_idx]['fp'] += fp
                    species_metrics[species_idx]['fn'] += fn

                for true_box in true_frame:
                    species_idx = true_box[0]
                    if not any(calculate_iou(pred_box, true_box) >= 0.5 and pred_box[0] == true_box[0] for pred_box in pred_frame):
                        species_metrics[species_idx]['fn'] += 1

        table_data = []

        for species_idx, metrics in species_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            species_name = self.species_names[species_idx] if self.species_names else f"Species {species_idx}"
            table_data.append([species_name, f"{precision:.2f}", f"{recall:.2f}", f"{f1_score:.2f}"])

        tp = background_metrics['tp']
        fp = background_metrics['fp']
        fn = background_metrics['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        table_data.append(["Background", f"{precision:.2f}", f"{recall:.2f}", f"{f1_score:.2f}"])

        headers = ["Species", "Precision", "Recall", "F1 Score"]
        total_tp = sum(metrics['tp'] for metrics in species_metrics.values()) + background_metrics['tp']
        total_fp = sum(metrics['fp'] for metrics in species_metrics.values()) + background_metrics['fp']
        total_fn = sum(metrics['fn'] for metrics in species_metrics.values()) + background_metrics['fn']

        total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        total_f1_score = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0

        table_data.append(["Total", f"{total_precision:.2f}", f"{total_recall:.2f}", f"{total_f1_score:.2f}"])
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    def run(self, test_dir):
        predicted_frames, true_frames, total_time = self.get_frames(test_dir)
        self.get_metrics(predicted_frames, true_frames)
        num_frames = len(os.listdir(os.path.join(test_dir, 'images')))
        avg_time_per_frame = total_time / num_frames

        print(f"\nTotal time: {total_time:.2f} seconds")
        print(f"Average time per frame: {avg_time_per_frame:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insect Classification using YOLO and ResNet")
    parser.add_argument('--data', type=str, required=True, help='Path to the test directory')
    parser.add_argument('--yolo_weights', type=str, required=True, help='Path to the YOLO model file')
    parser.add_argument('--resnet_weights', type=str, required=True, help='Path to the ResNet model file')
    parser.add_argument('--use_resnet50', action='store_true', help='Use ResNet50 instead of ResNet152')

    args = parser.parse_args()

    species_names = [
        "Coccinellidae septempunctata", "Apis mellifera", "Bombus lapidarius", "Bombus terrestris",
        "Eupeodes corolla", "Episyrphus balteatus", "Aglais urticae", "Vespula vulgaris",
        "Eristalis tenax", "Non-Bombus Anthophila", "Bombus spp.", "Syrphidae",
        "Fly spp.", "Unclear insect", "Mixed animals"
    ]

    classifier = TestTwoStage(args.yolo_weights, args.resnet_weights, use_resnet50=args.use_resnet50, species_names=species_names)
    classifier.run(args.data)