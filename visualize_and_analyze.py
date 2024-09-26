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

    # Load training dataset with transformations and without
    train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=train_transforms)
    train_dataset_no_transforms = datasets.ImageFolder(root=train_dataset_path)
    print(f"Number of training images: {len(train_dataset)}")

    # Visualize a few images from the training dataset (with no transforms already applied)
    visualize_data(train_dataset_no_transforms, num_images=5)

    # Visualize a few images from the training dataset (with transforms already applied)
    visualize_data(train_dataset, num_images=5)

    # Load validation dataset with transformations and without
    valid_dataset = datasets.ImageFolder(root=valid_dataset_path, transform=valid_transforms)
    valid_dataset_no_transforms = datasets.ImageFolder(root=valid_dataset_path)
    print(f"Number of validation images: {len(valid_dataset)}")

    # Visualize a few images from the validation dataset (with no transforms already applied)
    visualize_data(valid_dataset_no_transforms, num_images=5)

    # Visualize a few images from the validation dataset (with transforms already applied)
    #visualize_data(valid_dataset, num_images=5)

if __name__ == "__main__":
    main()
