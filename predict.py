import os

import torch
from PIL import Image
from torchvision.transforms import transforms

from models import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def predict():
    model = load_model('cnn')
    model.to(device)

    image = Image.open(os.path.join("data/New" + "/glioma_tumor.jpg"))

    # Create a variable representation of the ToTensor method.
    transform = transforms.ToTensor()
    # Convert PIL image to Pytorch tensor.
    tensorImg = transform(image)
    tensorImg = tensorImg.to(device)

    y_pred = model(tensorImg.unsqueeze(0))

    prediction = int(torch.max(y_pred.cpu().data, 1)[1].numpy())
    print(prediction)

    if (prediction == 0):
        print('Glioma')
    if (prediction == 1):
        print('Meningioma')
    if (prediction == 2):
        print('No Tumor')
    if (prediction == 3):
        print('Pituitary')


if __name__ == '__main__':
    predict()
