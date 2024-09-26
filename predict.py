import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from models import TransferLearningResNet  # Replace with the correct model architecture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the class names (should match the classes from training)
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Define image transformations (should match the ones used during training)
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure this matches your model's input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Function to load the model
def load_model(model_path):
    model = TransferLearningResNet(num_classes=len(class_names))  # Adjust architecture as needed
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


# Function to predict the class of a single image
def predict_image(model, image_path):
    image = Image.open(image_path).convert('RGB')  # Load the image and convert to RGB
    image = image_transforms(image)  # Apply the same transforms as training
    image = image.unsqueeze(0)  # Add batch dimension

    image = image.to(device)  # Move image to device
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    return predicted_class


# Function to display predictions for images in the "New" folder
def predict_new_images(model, folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        predicted_class = predict_image(model, image_path)

        # Display the image and its prediction
        image = Image.open(image_path)
        plt.imshow(np.array(image))
        plt.title(f'Predicted: {predicted_class}')
        plt.axis('off')
        plt.show()


# Main function to load model and predict new images
def main():
    # Load the trained model
    model_path = "resnet_model.pth"  # Path to your trained model
    model = load_model(model_path)

    # Path to the folder with new images
    new_images_folder = "data/New"

    # Predict and display the images
    predict_new_images(model, new_images_folder)


if __name__ == "__main__":
    main()
