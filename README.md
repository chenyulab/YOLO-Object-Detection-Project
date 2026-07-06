# YOLO-Object-Detection-Project
Code used for training and retraining custom YOLO models

This repository provides a full pipeline for training and running inference using a YOLO-based object detection model, including dataset preparation from CVAT exports and model evaluation utilities.

Built around Ultralytics YOLO (object detection framework).

------------------------------------------------------------
SETUP OVERVIEW
------------------------------------------------------------

This project includes:
- Dataset preprocessing (CVAT → YOLO format)
- YOLO model training
- Experiment tracking and saved weights
- Inference utilities for running predictions on new data

------------------------------------------------------------
1. CLONE THE REPOSITORY
------------------------------------------------------------

cd Documents/Github <br>
git clone https://github.com/chenyulab/YOLO-Object-Detection-Project.git <br>
cd YOLO-Object-Detection-Project <br>

------------------------------------------------------------
3. TRAIN A MODEL
------------------------------------------------------------

Edit scripts/train_model.py:

- model_path = path to base YOLO model (e.g. yolov8m.pt)
- dataset_path = path to CVAT dataset
- project_name = output directory for experiments
- training_run_name = experiment folder name

Optional:
Adjust hyperparameters in args_dict:
- epochs
- batch size
- mosaic
- patience

Run training:

python -m scripts.train_model

------------------------------------------------------------
4. DATASET PREPROCESSING (FLATTENING)
------------------------------------------------------------

If CVAT export contains nested folders, run:

flatten_dataset(dataset_path)

This:
- Moves images into images/
- Moves labels into labels/
- Matches .txt annotation files

If already flattened, you may comment it out.

------------------------------------------------------------
5. RUN INFERENCE
------------------------------------------------------------

After training:

Find model:
project_name/training_run_name/weights/best.pt

Set path in exp_utils.py:

model_path = "path/to/best.pt"

Run:

./run_obj_detection.sh

------------------------------------------------------------
TROUBLESHOOTING
------------------------------------------------------------

Error: No module named obj_detector <br>
→ run: <br>
python -m scripts.train_model <br>

Missing dependencies (yaml, torch, etc.) <br>
→ rerun: <br>
./setup.sh <br>

------------------------------------------------------------
RECOMMENDED WORKFLOW
------------------------------------------------------------

git pull <br>
./setup.sh <br>
source venv/Scripts/activate <br>
python -m scripts.train_model <br>

