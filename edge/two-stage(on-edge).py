import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.models import resnet152, resnet50
import torch.nn as nn
from picamera2 import Picamera2

class TwoStageEdge:
    def __init__(self, yolo_model, resnet_model, model_type='resnet152'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.yolo_model = yolo_model

        if model_type == 'resnet50':
            self.classification_model = resnet_model
        else:
            self.classification_model = resnet_model

        self.num_classes = 22
        self.classification_model.fc = nn.Linear(self.classification_model.fc.in_features, self.num_classes)
        state_dict = torch.load(f"species-{model_type}.pth", map_location=self.device)
        self.classification_model.load_state_dict(state_dict)
        self.classification_model.to(self.device)
        self.classification_model.eval()

        self.classification_transform = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.picam2 = Picamera2()
        camera_config = self.picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1080, 1080)})
        self.picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})
        self.picam2.configure(camera_config)
        self.picam2.start()

    def frame_processing(self, species_names, classification_counts):
        frame = self.picam2.capture_array()
        results = self.yolo_model.track(source=frame, conf=0.3, iou=0.5, tracker='botsort.yaml', persist=True)
        frame = results[0].orig_img
        detections = results[0].boxes

        for box in detections:
            try:
                xyxy = box.xyxy.cpu().numpy().flatten().astype(int) # get the coordinates of the box
                x1, y1, x2, y2 = xyxy[:4] # get the coordinates of the box
            except Exception as e:
                print(f"Error processing box coordinates: {e}")
                continue

            h, w = frame.shape[:2]
            track_id = getattr(box, 'id', None)
            if track_id is not None:
                try:
                    track_id = int(track_id.item() if hasattr(track_id, 'item') else track_id)
                except Exception as e:
                    track_id = "N/A"
            else:
                track_id = "N/A"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            insect_crop = frame[y1:y2, x1:x2]
            insect_crop_rgb = cv2.cvtColor(insect_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(insect_crop_rgb)
            input_tensor = self.classification_transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.classification_model(input_tensor)

            predicted_class_idx = outputs.argmax(dim=1).item()
            species_name = species_names[predicted_class_idx] if predicted_class_idx < len(species_names) else "Unknown"
            if track_id not in classification_counts:
                classification_counts[track_id] = {}
            if species_name not in classification_counts[track_id]:
                classification_counts[track_id][species_name] = 0
            classification_counts[track_id][species_name] += 1

            label = f"ID: {track_id} | {species_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            org = (x1, y1 - 10)
            cv2.putText(frame, label, org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, label, org, font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        cv2.imshow("Insect Tracking & Classification", frame)

        return classification_counts

    def run(self):
        species_names = [
            "Aglais io", "Aglais urticae", "Bombus hortorum", "Bombus lapidarius",
            "Bombus lucorum", "Bombus monticola", "Bombus pascuorum", "Colias croceus",
            "Colletes hederae", "Episyrphus balteatus", "Eristalis tenax", "Gonepteryx rhamni",
            "Myathropa florea", "Pieris brassicae", "Pieris rapae", "Polygonia c-album",
            "Rhingia campestris", "Syrphus ribesii", "Vanessa atalanta", "Vanessa cardui",
            "Vespa crabro", "Vespula vulgaris"
        ]

        while True:
            classification_counts = {}
            classification_counts = self.frame_processing(species_names, classification_counts)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(classification_counts)
        cv2.destroyAllWindows()
        self.picam2.stop()

if __name__ == "__main__":
    classifier = TwoStageEdge(yolo_model=YOLO("insect-yolov8.pt"), resnet_model='species_resnet152.pth', model_type='resnet152')  # Change to 'resnet50' if needed
    classifier.run()