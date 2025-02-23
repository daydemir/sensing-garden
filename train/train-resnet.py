#pip install torch torchvision scikit-learn matplotlib tqdm torchmetrics pillow
#python3 train-resnet.py --data_dir '' --output_dir '' --arch resnet50 --img_size 956 --num_epochs 50 --batch_size 4

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import os
import copy
from PIL import Image
import logging
from torchvision.models import ResNet152_Weights, ResNet50_Weights
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchmetrics
from collections import Counter
import argparse
from datetime import datetime
import json

def parse_arguments():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train ResNet Classifier for Insect Detection')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory containing "train" and "val" subdirectories')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save logs, graphs, metrics, and models')
    parser.add_argument('--arch', type=str, default='resnet152', choices=['resnet152', 'resnet50'],
                        help='Model architecture to use: resnet152 or resnet50')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='Image resolution for training and validation')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs to train the model')
    args = parser.parse_args()
    return args

def setup_directories(output_dir):
    """Create necessary directories for logs, graphs, metrics, and models."""
    logs_dir = os.path.join(output_dir, 'logs')  # Directory for logs
    graphs_dir = os.path.join(output_dir, 'graphs')  # Directory for graphs
    metrics_dir = os.path.join(output_dir, 'metrics')  # Directory for metrics
    models_dir = os.path.join(output_dir, 'models')  # Directory for models
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    return logs_dir, graphs_dir, metrics_dir, models_dir

def get_transforms(args):
    """Return training and validation transforms based on the desired image size."""
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size),  # Crop image randomly and resize
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomRotation(15),  # Random rotation within 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5),  # Random Gaussian blur with 50% probability
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize (imagenet stats)
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((int(args.img_size * 1.2), int(args.img_size * 1.2))),  # Resize image to 1.2 times the specified size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transforms, val_transforms

class SafeImageFolder(datasets.ImageFolder):
    """Custom ImageFolder that skips corrupted images."""

    def __init__(self, root, transform=None, loader=datasets.folder.default_loader, is_valid_file=None):
        """Initialize the dataset and filter out corrupted images."""
        super(SafeImageFolder, self).__init__(root, transform=transform, loader=loader, is_valid_file=is_valid_file)
        self.valid_indices = []
        for idx, (path, _) in enumerate(self.samples):
            try:
                with Image.open(path) as img:
                    img.verify()  # Verify image integrity
                self.valid_indices.append(idx)  # Add index if image is valid
            except Exception as e:
                logging.warning(f"Skipping corrupted image {path}: {e}")  # Log corrupt image
        self.samples = [self.samples[i] for i in self.valid_indices]  # Filter samples
        self.targets = [self.targets[i] for i in self.valid_indices]  # Filter targets

    def __getitem__(self, index):
        """Retrieve an image and its target; if corrupted, recursively fetch the next."""
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target
        except Exception as e:
            logging.warning(f"Skipping corrupted image {path} during training: {e}")  # Log error during loading
            return self.__getitem__((index + 1) % len(self.samples))  # Try next image

def train_model(model, criterion, optimizer, scheduler, num_epochs, device, train_loader, val_loader, dataset_sizes, class_names, logs_dir, graphs_dir, metrics_dir):
    """Train the model and return the best model based on validation accuracy."""
    since = time.time()  # Record start time
    best_model_wts = copy.deepcopy(model.state_dict())  # Store initial model state  # best weights so far
    best_acc = 0.0  # Initialize best accuracy
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names)).to(device)  # Initialize accuracy metric  # complex metric init
    val_precision = torchmetrics.Precision(task='multiclass', average='macro', num_classes=len(class_names)).to(device)  # Initialize precision metric
    val_recall = torchmetrics.Recall(task='multiclass', average='macro', num_classes=len(class_names)).to(device)  # Initialize recall metric
    val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=len(class_names)).to(device)  # Initialize IoU metric
    val_map = torchmetrics.AveragePrecision(task='multiclass', num_classes=len(class_names), average='macro').to(device)  # Initialize mAP metric
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')  # Display current epoch
        print('-' * 10)  # Separator line
        val_accuracy.reset()  # Reset metric for new epoch
        val_precision.reset()  # Reset metric
        val_recall.reset()  # Reset metric
        val_iou.reset()  # Reset metric
        val_map.reset()  # Reset metric
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                loader = train_loader
            else:
                model.eval()  # Set model to evaluation mode
                loader = val_loader
            running_loss = 0.0  # Initialize loss accumulator
            running_corrects = 0  # Initialize correct predictions accumulator
            all_preds = []  # Store all predictions for metrics
            all_labels = []  # Store all true labels for metrics
            pbar = tqdm(loader, desc=f'{phase.capitalize()} Phase', unit='batch')  # Initialize progress bar  # complex f-string-based progress bar
            for inputs, labels in pbar:
                if inputs is None:
                    continue  # Skip if batch is empty
                inputs = inputs.to(device)  # Move inputs to device  # complex device transfer
                labels = labels.to(device)  # Move labels to device
                optimizer.zero_grad()  # Reset gradients
                with torch.set_grad_enabled(phase == 'train'):  # Enable gradients only in train phase
                    with torch.cuda.amp.autocast():  # Enable mixed precision
                        outputs = model(inputs)  # Forward pass
                        _, preds = torch.max(outputs, 1)  # Get predictions  # complex tensor operation
                        loss = criterion(outputs, labels)  # Compute loss
                    if phase == 'train':
                        scaler.scale(loss).backward()  # Backpropagation with scaling
                        scaler.step(optimizer)  # Update parameters
                        scaler.update()  # Update scaler for mixed precision
                running_loss += loss.item() * inputs.size(0)  # Accumulate loss (scaled by batch size)
                running_corrects += torch.sum(preds == labels.data)  # Accumulate correct predictions  # tensor sum operation
                all_preds.extend(preds.cpu().numpy())  # Store predictions (move to CPU and convert to numpy)
                all_labels.extend(labels.cpu().numpy())  # Store true labels
                batch_loss = loss.item()  # Extract loss value
                batch_acc = torch.sum(preds == labels.data).double().item() / len(labels)  # Compute batch accuracy  # division after type conversion
                pbar.set_postfix({'Loss': f'{batch_loss:.4f}', 'Acc': f'{batch_acc:.4f}'})  # Update progress bar with metrics
                if phase == 'val':
                    val_accuracy.update(preds, labels)  # Update accuracy metric
                    val_precision.update(preds, labels)  # Update precision metric
                    val_recall.update(preds, labels)  # Update recall metric
                    val_iou.update(preds, labels)  # Update IoU metric
                    val_map.update(outputs, labels)  # Update mAP metric
            if phase == 'train':
                scheduler.step()  # Update learning rate scheduler
            epoch_loss = running_loss / dataset_sizes[phase]  # Compute average loss for epoch
            epoch_acc = running_corrects.double() / dataset_sizes[phase]  # Compute average accuracy for epoch
            if phase == 'val':
                cls_report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)  # Generate classification report
                conf_matrix = confusion_matrix(all_labels, all_preds)  # Generate confusion matrix
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')  # Print validation loss and accuracy
                print(f'{phase.capitalize()} Classification Report:\n{cls_report}')  # Print detailed classification report
                plt.figure(figsize=(12, 10))  # Create a new figure
                plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)  # Display confusion matrix with color mapping
                plt.title('Confusion Matrix')  # Set plot title
                plt.colorbar()  # Show color bar
                tick_marks = np.arange(len(class_names))  # Compute tick marks for classes  # complex index generation
                plt.xticks(tick_marks, class_names, rotation=90)  # Set x-axis tick labels
                plt.yticks(tick_marks, class_names)  # Set y-axis tick labels
                fmt = 'd'
                thresh = conf_matrix.max() / 2.  # Calculate threshold for text color contrast  # complex computation
                for i, j in np.ndindex(conf_matrix.shape):
                    plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment="center", color="white" if conf_matrix[i, j] > thresh else "black")  # Annotate cells in the matrix
                plt.ylabel('True Label')  # Set y-axis label
                plt.xlabel('Predicted Label')  # Set x-axis label
                plt.tight_layout()  # Adjust layout
                plt.savefig(os.path.join(graphs_dir, f'confusion_matrix_epoch_{epoch+1}.png'))  # Save confusion matrix figure
                plt.close()  # Close the figure to free memory
                with open(os.path.join(metrics_dir, 'classification_reports.txt'), 'a') as f:
                    f.write(f'Epoch {epoch+1} Classification Report:\n')
                    f.write(cls_report)
                    f.write('\n')
                val_current_accuracy = val_accuracy.compute().item()  # Compute final accuracy for validation phase  # complex metric finalization
                val_current_precision = val_precision.compute().item()  # Compute final precision
                val_current_recall = val_recall.compute().item()  # Compute final recall
                val_current_iou = val_iou.compute().item()  # Compute final IoU
                val_current_map = val_map.compute().item()  # Compute final mAP
                print(f'Validation Metrics:')
                print(f'  Accuracy: {val_current_accuracy:.4f}')
                print(f'  Precision: {val_current_precision:.4f}')
                print(f'  Recall: {val_current_recall:.4f}')
                print(f'  IoU (Jaccard Index): {val_current_iou:.4f}')
                print(f'  mAP@0.5: {val_current_map:.4f}')
                with open(os.path.join(metrics_dir, 'validation_metrics.log'), 'a') as f:
                    f.write(f'Epoch {epoch+1} Validation Metrics:\n')
                    f.write(f'  Accuracy: {val_current_accuracy:.4f}\n')
                    f.write(f'  Precision: {val_current_precision:.4f}\n')
                    f.write(f'  Recall: {val_current_recall:.4f}\n')
                    f.write(f'  IoU (Jaccard Index): {val_current_iou:.4f}\n')
                    f.write(f'  mAP@0.5: {val_current_map:.4f}\n\n')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc  # Update best accuracy
                best_model_wts = copy.deepcopy(model.state_dict())  # Save best model weights  # deep copy for safety
        print()
        torch.cuda.empty_cache()  # Clear GPU cache after each epoch
    time_elapsed = time.time() - since  # Calculate elapsed time
    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)  # Load best model weights
    return model

def main():
    """Main entry point for training the model."""
    args = parse_arguments()  # Parse command-line arguments
    output_dir = args.output_dir
    logs_dir, graphs_dir, metrics_dir, models_dir = setup_directories(output_dir)  # Setup directories
    data_dir = args.data_dir 
    train_dir = os.path.join(data_dir, 'train')  # Directory for training data
    val_dir = os.path.join(data_dir, 'val')  # Directory for validation data
    train_transforms, val_transforms = get_transforms(args)  # Get image transforms
    logging.basicConfig(filename=os.path.join(logs_dir, 'data_loading.log'), level=logging.WARNING, format='%(asctime)s:%(levelname)s:%(message)s')  # Setup logging
    train_dataset = SafeImageFolder(train_dir, transform=train_transforms)  # Create training dataset
    val_dataset = SafeImageFolder(val_dir, transform=val_transforms)  # Create validation dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)  # Create training loader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)  # Create validation loader
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}  # Record dataset sizes
    class_names = train_dataset.classes  # Get class names from dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Select device
    if args.arch == 'resnet152':
        model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)  # Load ResNet152 weights
    else:
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Load ResNet50 weights
    num_ftrs = model.fc.in_features  # Get number of input features from final layer  # complex layer query
    model.fc = nn.Linear(num_ftrs, len(class_names))  # Replace classifier layer with new linear layer
    model = model.to(device)  # Move model to device  # device transfer
    criterion = nn.CrossEntropyLoss()  # Define loss function
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Initialize optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Initialize learning rate scheduler
    train_counts = Counter(train_dataset.targets)  # Count training set labels
    print("Training set class distribution:")
    for idx, count in train_counts.items():
        print(f"{class_names[idx]}: {count}")  # Print training distribution
    val_counts = Counter(val_dataset.targets)  # Count validation set labels
    print("\nValidation set class distribution:")
    for idx, count in val_counts.items():
        print(f"{class_names[idx]}: {count}")  # Print validation distribution
    model = train_model(model, criterion, optimizer, scheduler, args.num_epochs, device, train_loader, val_loader, dataset_sizes, class_names, logs_dir, graphs_dir, metrics_dir)  # Train the model
    torch.save(model.state_dict(), os.path.join(models_dir, 'insect_resnet152.pth'))  # Save model weights
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Generate a timestamp  # complex date formatting
    new_output_dir = os.path.join(args.output_dir, f'run_{timestamp}')  # Create a new run directory
    os.makedirs(new_output_dir, exist_ok=True)
    hyperparams = {
        'batch_size': args.batch_size,
        'learning_rate': 0.0001,
        'num_epochs': args.num_epochs,
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'model': args.arch
    }
    with open(os.path.join(new_output_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=4)  # Save hyperparameters to JSON

if __name__ == "__main__":
    main()

