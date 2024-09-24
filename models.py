import torch
from torch import nn
from torchvision import models
import timm

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNClassifier, self).__init__()

        # Adding more convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adaptive pooling to allow variable input image sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Fully connected layers with Dropout to prevent overfitting
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x


# Pretrained model option (ResNet) using transfer learning
# Pretrained model option (ResNet) using transfer learning
import torch
from torch import nn
from torchvision import models

class TransferLearningResNet(nn.Module):
    def __init__(self, num_classes=4, dropout_rate_1=0.3, dropout_rate_2=0.25):
        super(TransferLearningResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        # Freeze all layers except the last block (layer4)
        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze all layers

        # Unfreeze layer3 for fine-tuning
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True

        # Unfreeze layer4 for fine-tuning
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # Modify the final fully connected layers
        self.resnet.fc = nn.Sequential(
            nn.Flatten(),  # Flatten the output of the convolutional layers
            nn.Dropout(dropout_rate_1),  # First dropout layer
            nn.Linear(self.resnet.fc.in_features, 128),  # Dense layer with 128 units
            nn.ReLU(),  # ReLU activation
            nn.Dropout(dropout_rate_2),  # Second dropout layer
            nn.Linear(128, num_classes),  # Final dense layer with output matching number of classes
            nn.Softmax(dim=1)  # Softmax activation for multi-class classification
        )

    def forward(self, x):
        return self.resnet(x)


class TransferLearningXception(nn.Module):
    def __init__(self, num_classes=4, img_size=299, dropout_rate_1=0.3, dropout_rate_2=0.25):
        super(TransferLearningXception, self).__init__()

        # Load the pre-trained Xception model from timm, exclude the top classification layer
        self.base_model = timm.create_model('xception', pretrained=True, num_classes=0, global_pool='max')

        # Add additional layers (like dropout and fully connected layers)
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout_rate_1),  # First dropout layer
            nn.Linear(self.base_model.num_features, 128),  # Dense layer with 128 units
            nn.ReLU(),  # ReLU activation
            nn.Dropout(dropout_rate_2),  # Second dropout layer
            nn.Linear(128, num_classes),  # Final dense layer for classification
            nn.Softmax(dim=1)  # Softmax for multi-class classification
        )

    def forward(self, x):
        x = self.base_model(x)  # Pass through the base Xception model
        x = self.fc_layers(x)  # Pass through the fully connected layers
        return x


# Model factory updated to include both CNN and transfer learning options
model_factory = {
    'improved_cnn': CNNClassifier,
    'resnet': TransferLearningResNet,
    'xception': TransferLearningXception
}


# Option to save the model (unchanged from original)
def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            save(model.state_dict(), path.join('./', f"{n}_model.pth"))
            print(f"Model saved as {n}_model.pth")
