from pathlib import Path

from brain_tumor_classification.inference import display_prediction, predict_folder

if __name__ == "__main__":
    checkpoint_path = Path("./resnet_model.pth")
    folder_path = Path("data/New")
    for result in predict_folder(checkpoint_path, folder_path=folder_path):
        image_name = Path(result["image_path"]).name
        predicted_class = str(result["display_class"])
        print(f"Image: {image_name}, Predicted class: {predicted_class}")
        display_prediction(Path(result["image_path"]), predicted_class)
