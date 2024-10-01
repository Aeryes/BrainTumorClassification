# Brain Tumor Classification
I attempt to solve the problem found here --> https://www.kaggle.com/datasets/prathamgrover/brain-tumor-classification/data

## Introduction:
This repo contains the code required to train a ResNet50 pretrained model to identify brain tumors based on the Kaggle dataset above.
The model was trained to 98.77% accuracy.

## Method
I attempted to use the legacy Xception model from the timm package at first, but it was not providing me with the results that I wanted to see. I swapped
over to a ResNet50 and added in data augmentation to help prevent overfitting. I noticed overfitting before adding in the data
augmentations. 

I added in a scheduler to adjust the learning rate as training progressed which helped the model generalize better. Early stopping is also in the scripts
but was not needed due to the small epoch number required to reach a good end result.

I also performed some visualizations of the data to see what I was working with before training and to test my ideas for data augmentation.

### Here are 5 samples of the training data with no data augmentation:
![alt text](/images/train_data_no_transforms.png "Traning Data No Transforms")

### Here are 5 samples of the training data with data augmentation:
![alt text](/images/train_data_with_transforms.png "Traning Data With Transforms")

### Here are 5 samples of the validation data with no data augmentation:
![alt text](/images/validation_data_no_transforms.png "Validation Data No Transforms")

## Results
### Here is the training over 16 epochs:
![alt text](/images/Figure_1.png "Training Results")

### Here is the confusion matrix of the trained model:
![alt text](/images/confusion_matrix.png "Confusion Matrix")

### Predicted Outputs
Both of the new images were Glioma tumors and were predicted correctly.
![alt text](/images/predicted_one.png "Predicted Image One")

![alt text](/images/predicted_two.png "Predicted Image Two")


## How to use?
Use the following:

```git clone https://github.com/Aeryes/BrainTumorClassification.git```

Run ```pip install -r requirements.txt```

Place your new images in the data/New folder.

Run ```python predict.py```

