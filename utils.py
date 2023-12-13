from PIL import Image
import csv
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

LABEL_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


class DataSet(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.tensorImg = ""
        self.csvDict = {}
        self.tensorList = []
        self.counter = 0
        self.labelType = ""
        self.csvDict = {}
        self.transform_two = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        self.transform_three = transforms.RandomHorizontalFlip()
        self.transform_four = transforms.RandomCrop(40)
        self.transform_five = transforms.Resize((256, 256))

        # Add glioma_tumor images to the csvDict.
        for i, filename in enumerate(os.listdir(dataset_path + "/glioma_tumor/")):
            f = os.path.join(dataset_path + "/glioma_tumor/" + filename)
            image = Image.open(f)
            #image.show()

            # Resize each image so they are all equal in size.
            image = self.transform_five(image)
            #image = self.transform_two(image)
            #image = self.transform_three(image)

            # Create a variable representation of the ToTensor method.
            transform = transforms.ToTensor()
            # Convert PIL image to Pytorch tensor.
            self.tensorImg = transform(image)

            self.tensorList.append((self.tensorImg, 0))

            # Increment the counter.
            self.counter += 1


        # Add meningioma_tumor images to the csvDict.
        for i, filename in enumerate(os.listdir(dataset_path + "/meningioma_tumor/")):
            f = os.path.join(dataset_path + "/meningioma_tumor/" + filename)
            image = Image.open(f)
            #image.show()

            # Resize each image so they are all equal in size.
            image = self.transform_five(image)

            # Create a variable representation of the ToTensor method.
            transform = transforms.ToTensor()
            # Convert PIL image to Pytorch tensor.
            self.tensorImg = transform(image)

            self.tensorList.append((self.tensorImg, 1))

            # Increment the counter.
            self.counter += 1
        # Add no_tumor images to the csvDict.
        for i, filename in enumerate(os.listdir(dataset_path + "/no_tumor/")):
            f = os.path.join(dataset_path + "/no_tumor/" + filename)
            image = Image.open(f)
            #image.show()

            # Resize each image so they are all equal in size.
            image = self.transform_five(image)

            # Create a variable representation of the ToTensor method.
            transform = transforms.ToTensor()
            # Convert PIL image to Pytorch tensor.
            self.tensorImg = transform(image)

            self.tensorList.append((self.tensorImg, 2))

            # Increment the counter.
            self.counter += 1

        # Add pituitary_tumor images to the csvDict.
        for i, filename in enumerate(os.listdir(dataset_path + "/pituitary_tumor/")):
            f = os.path.join(dataset_path + "/pituitary_tumor/" + filename)
            image = Image.open(f)
            #image.show()

            # Resize each image so they are all equal in size.
            image = self.transform_five(image)

            # Create a variable representation of the ToTensor method.
            transform = transforms.ToTensor()
            # Convert PIL image to Pytorch tensor.
            self.tensorImg = transform(image)

            self.tensorList.append((self.tensorImg, 3))

            # Increment the counter.
            self.counter += 1

        print("Total amount of data points: " + str(len(self.tensorList)))

    def __len__(self):
        # _, _, files = next(os.walk(self.dataset_path))
        # length = len(files)
        # return length - 1
        return len(self.tensorList)

    def __getitem__(self, idx):
        return self.tensorList[idx]

def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    dataset = DataSet(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)
