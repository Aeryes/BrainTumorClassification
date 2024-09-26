
from PIL import Image
import os
from torch.utils.data import Dataset

LABEL_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

class BrainTumorDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.data = []

        # Load images and labels
        for label_idx, label_name in enumerate(LABEL_NAMES):
            label_path = os.path.join(dataset_path, label_name)
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                self.data.append((img_path, label_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(dataset_path, transform=None):
    return BrainTumorDataset(dataset_path, transform=transform)
