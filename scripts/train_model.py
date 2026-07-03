import obj_detector.constants as constants
from pathlib import Path
from obj_detector.trainer import Trainer  # Importing the PredictorModel class from your predictor module
import shutil
from tqdm import tqdm

'''
Created by Jacob Rivera
Spring 2024

Last edit: 2/28/2025

Description:
    Train YOLO model from passed arguements

    model_path: 
        model to train. 
        if continuing to train a model, path should be to the .pt file
    
    dataset_path:
        path to the top level directory of your dataset
        in the visual below, I would pass the path to "input_dir"
        
        dataset_path/
            ├── images/
            │   ├── file1.jpg
            │   ├── file2.jpg
            │   ├── file3.jpg
            │   └── file4.jpg
            └── labels/
                ├── file1.txt
                ├── file2.txt
                ├── file3.txt
                └── file4.txt

        data_output_dir/
            ├── train/
            |   ├── images/
            |   |   ├── file1.jpg
            |   │   └── file2.jpg
            |   └── labels/
            |       ├── file1.txt
            |       └── file2.txt
            └── test/
                ├── images/
                |   ├── file3.jpg
                │   └── file4.jpg
                └── labels/
                    ├── file3.txt
                    └── file4.txt

    class_dict:
        a python dictionary containing {int:"class_label"} pairs
        the ints must start from 0 and each label should be unique
        a dict can be passed in or set in the constants file

'''


model_path = Path(r"C:\Users\multimaster\Documents\Github\YOLO-Object-Detection-Project\data\base_models\yolov8m.pt")
# model_path = Path(r"C:\Users\multimaster\documents\YOLO-Object-Detection-Project\yolo26n.pt")

dataset_path = Path(r"Z:\James\YOLO310_JW\FullCVAT_7_3")
class_dict = constants.CLASSES_DICT

project_name = r"C:\Users\multimaster\Documents\YOLO_310\models" # name of the project for various trainings, example: "experiment_12"
training_run_name = "310_CVAT_JW_7_3" # name of individual training for a project, example: "train1"

def flatten_dataset(dataset_path: Path):
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    img_exts = [".jpg", ".jpeg", ".png", ".bmp"]

    all_images = [p for p in images_dir.rglob("*") if p.suffix.lower() in img_exts]

    moved = 0

    for img_path in tqdm(all_images, desc="Flattening dataset", unit="img"):
        if img_path.parent == images_dir:
            continue

        new_img_path = images_dir / img_path.name
        if new_img_path.exists():
            continue


        rel_path = img_path.relative_to(images_dir)
        old_label_path = (labels_dir / rel_path).with_suffix(".txt")
        new_label_path = labels_dir / (img_path.stem + ".txt")

        shutil.move(str(img_path), str(new_img_path))

        if old_label_path.exists():
            if not new_label_path.exists():
                shutil.move(str(old_label_path), str(new_label_path))
        else:
            print(f"Missing label for: {img_path.name}")

        moved += 1
    
    print(f"Dataset flattened. Moved {moved} files.")
def main():

    flatten_dataset(dataset_path)

    args_dict = {
        "model": model_path,
        "epochs": 200,
        "device": 0, # set to 'cpu' or delete if no GPU
        "project": project_name,
        "name": training_run_name,
        "mosaic": 1,
        "patience": 30,
        "batch": 16,
        "close_mosaic": 50,
        "iou": 0.7,
        "scale": 0.5, 
        "box": 7.5
    }

    trainer = Trainer(args=args_dict, dataset_path=dataset_path, class_dict=class_dict, split_data_save_path=dataset_path)
    trainer.dataMaster.generate_mirror_vars(put_back=True)
    
    trainer.train(show_output=True)
    
    return



if __name__ == "__main__":
    main()