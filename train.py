from utils import load_data
from models import CNNClassifier, save_model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train(args):
    model = CNNClassifier()
    model.to(device)

    learning_rate = 0.001
    epoch_val = 30
    optimizer = torch.optim.Adam(model.parameters())
    loss = torch.nn.CrossEntropyLoss()

    # Load the train data.
    data_train = load_data("data/Training")

    # Validation data.
    data_valid = load_data("data/Testing")

    # Training.
    for epoch in range(epoch_val):
        loss_train = []
        loss_valid = []
        num_correct_train = 0
        num_predictions_train = 0
        num_correct_valid = 0
        num_predictions_valid = 0

        for train_features, train_labels in data_train:
            model.train()
            optimizer.zero_grad()

            train_features = torch.tensor(train_features, device=device)
            train_labels = torch.tensor(train_labels, device=device)

            y_pred = model(train_features)
            actualLoss = loss(y_pred, train_labels)

            _, predicted = torch.max(y_pred.data, 1)
            num_correct_train += (predicted == train_labels).sum().item()
            num_predictions_train += train_labels.size(0)

            loss_train.append(actualLoss.detach().cpu().numpy())

            actualLoss.backward()
            optimizer.step()

        for valid_features, valid_labels in data_valid:
            model.eval()
            running_loss_valid = 0

            valid_features = torch.tensor(valid_features, device=device)
            valid_labels = torch.tensor(valid_labels, device=device)

            # Validaton data
            with torch.no_grad():
                y_pred = model(valid_features)
                actualLoss = loss(y_pred, valid_labels)
                running_loss_valid += actualLoss.item()
                loss_valid.append(actualLoss.detach().cpu().numpy())

                _, predicted = torch.max(y_pred.data, 1)
                num_correct_valid += (predicted == valid_labels).sum().item()
                num_predictions_valid += valid_labels.size(0)

        avg_train_loss = sum(loss_train) / len(loss_train)
        avg_train_acc = num_correct_train / num_predictions_train * 100

        avg_accuracy_valid = num_correct_valid / num_predictions_valid * 100
        avg_valid_loss = sum(loss_valid) / len(loss_valid)

        print("[Epoch: %d / %d],  [Train loss: %.4f], [Acc train: %.2f],  [Valid loss: %.4f],  [Acc valid: %.2f]" \
              % (epoch + 1, epoch_val, avg_train_loss, avg_train_acc, avg_valid_loss, avg_accuracy_valid))

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
