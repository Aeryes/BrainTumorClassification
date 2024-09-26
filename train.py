from utils import load_data
from sklearn.utils.class_weight import compute_class_weight
from models import save_model, TransferLearningResNet
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm  # For live progress updates
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered")
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train(args):
    model = TransferLearningResNet(num_classes=4, dropout_rate_1=0.5, dropout_rate_2=0.5)
    model.to(device)

    # Define transforms for data augmentation and normalization
    from torchvision import transforms
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

    # Load the train and validation data.
    train_dataset = load_data("data/Training", transform=train_transforms)
    valid_dataset = load_data("data/Testing", transform=valid_transforms)

    # Dataloaders for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Assuming you have the labels available from your dataset
    train_labels = [label for _, label in train_dataset]  # Iterate directly over train_dataset

    # Compute class weights based on the training dataset's label distribution
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)

    # Convert to tensor and move to device
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Optimizer: fine-tune the final layer and layer4
    optimizer = torch.optim.Adam([
        #{'params': model.resnet.layer3.parameters(), 'lr': 1e-5}, # layer3
        {'params': model.resnet.layer4.parameters(), 'lr': 1e-5},  # layer4
        {'params': model.resnet.fc.parameters(), 'lr': 1e-4}  # layer4
    ], weight_decay=1e-4)

    # Use class weights in the loss function
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Learning rate scheduler (reduce LR when validation loss plateaus)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Early stopping to prevent overfitting
    #early_stopping = EarlyStopping(patience=2, min_delta=0.001)

    # Lists for storing metrics
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    # Set up plot
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training loop
    for epoch in range(16):
        # Train the model
        model.train()
        running_train_loss = 0.0
        num_correct_train = 0
        num_predictions_train = 0

        for train_features, train_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/100 - Training"):
            train_features, train_labels = train_features.to(device), train_labels.to(device)

            optimizer.zero_grad()
            y_pred = model(train_features)
            loss = loss_fn(y_pred, train_labels)

            loss.backward()
            optimizer.step()

            # Metrics tracking
            running_train_loss += loss.item()
            _, predicted = torch.max(y_pred, 1)
            num_correct_train += (predicted == train_labels).sum().item()
            num_predictions_train += train_labels.size(0)

        # Compute average metrics
        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_acc = num_correct_train / num_predictions_train * 100
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)

        # Validate the model
        model.eval()
        running_valid_loss = 0.0
        num_correct_valid = 0
        num_predictions_valid = 0

        with torch.no_grad():
            for valid_features, valid_labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/100 - Validating"):
                valid_features, valid_labels = valid_features.to(device), valid_labels.to(device)

                y_pred = model(valid_features)
                loss = loss_fn(y_pred, valid_labels)

                running_valid_loss += loss.item()
                _, predicted = torch.max(y_pred, 1)
                num_correct_valid += (predicted == valid_labels).sum().item()
                num_predictions_valid += valid_labels.size(0)

        avg_valid_loss = running_valid_loss / len(valid_loader)
        avg_valid_acc = num_correct_valid / num_predictions_valid * 100
        valid_losses.append(avg_valid_loss)
        valid_accuracies.append(avg_valid_acc)

        print(f"[Epoch {epoch+1}/100] Train loss: {avg_train_loss:.4f}, Train acc: {avg_train_acc:.2f}%, "
              f"Valid loss: {avg_valid_loss:.4f}, Valid acc: {avg_valid_acc:.2f}%")

        # Step the learning rate scheduler
        scheduler.step(avg_valid_loss)

        # Early stopping check
        #early_stopping(avg_valid_loss)
        #if early_stopping.early_stop:
        #    print("Early stopping triggered.")
        #    break

        # Live update of the graph
        ax1.clear()
        ax2.clear()

        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(valid_losses, label='Valid Loss')
        ax1.set_title('Loss over Epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(train_accuracies, label='Train Accuracy')
        ax2.plot(valid_accuracies, label='Valid Accuracy')
        ax2.set_title('Accuracy over Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()

        plt.pause(0.01)  # Update the graph in real-time

    save_model(model)
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show final plot


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    args = parser.parse_args()
    train(args)
