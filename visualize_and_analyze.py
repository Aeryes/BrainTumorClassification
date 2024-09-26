# Thanks to Tim Dangeon on Kaggle --> https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/discussion/482896 for the remove duplicate images code.

import os
import hashlib
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from models import TransferLearningResNet  # Import your model architecture here

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to compute hash for detecting duplicates
def compute_hash(file):
    hasher = hashlib.md5()
    with open(file, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Function to list files and detect duplicates
def list_files(hash_dict, dataset_path):
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            if file.endswith(".jpg"):
                file_hash = compute_hash(file_path)
                if file_hash in hash_dict:
                    hash_dict[file_hash].append(file_path)
                else:
                    hash_dict[file_hash] = [file_path]

# Function to remove duplicate images
def remove_duplicates(hash_dict):
    duplicate_count = 0
    for hash_value, file_paths in hash_dict.items():
        if len(file_paths) > 1:
            for file_path in file_paths[1:]:
                print(f"Removing duplicate (hash: {hash_value}): {file_path}")
                os.remove(file_path)
                duplicate_count += 1
    print(f"Number of duplicates removed: {duplicate_count}")

# Function to visualize a few images from the dataset
def visualize_data(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for i in range(num_images):
        image, label = dataset[i]

        if isinstance(image, torch.Tensor):
            image_np = image.numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        else:
            image_np = np.array(image)  # Convert PIL image to NumPy array

        axes[i].imshow(image_np)
        axes[i].axis('off')
        axes[i].set_title(f'Label: {dataset.classes[label]}')

    plt.show()

# Function to evaluate model and generate confusion matrix
def evaluate_model(model, loader, class_names):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

# Main function for loading data, removing duplicates, and visualizing
def main():
    # Define the transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets with transformations
    train_dataset_path = "data/Training"
    valid_dataset_path = "data/Testing"

    # Load training dataset with transformations
    train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=train_transforms)
    print(f"Number of training images: {len(train_dataset)}")

    # Load validation dataset with transformations
    valid_dataset = datasets.ImageFolder(root=valid_dataset_path, transform=valid_transforms)
    print(f"Number of validation images: {len(valid_dataset)}")

    # Create DataLoader for validation dataset
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Create the model architecture and load the state dict
    model = TransferLearningResNet(num_classes=4)  # Use the correct architecture that was used during training
    model.load_state_dict(torch.load("resnet_model.pth"))  # Load the state dict (weights)
    model.to(device)  # Move model to GPU if available

    # Evaluate the model and display confusion matrix
    class_names = valid_dataset.classes  # Get the class names
    evaluate_model(model, valid_loader, class_names)

if __name__ == "__main__":
    main()
