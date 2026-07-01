from pathlib import Path
from typing import Dict, Any
from obj_detector.data import DataMaster
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

class Trainer():
    def __init__(self, args: dict[str, Any], dataset_path, class_dict: dict[int, str] = None, split_data_save_path=None):
        self.model_path = args['model'] if 'model' in args else 'yolov8s.pt'
        self.args = args
        self.dataset_path = dataset_path
        self.seed = 32
        self.save_path = split_data_save_path
        self.dataMaster = DataMaster(self.dataset_path, class_dict, save_path=self.save_path)
        self.model = None

    def load_model(self):
        # Ensure the model is in the passed aruements
        self.args["model"] =  self.model_path
        self.model = DetectionTrainer(overrides=self.args)
        return

    def get_yaml(self):
        self.yaml = self.dataMaster.yaml
        self.args['data'] = self.yaml

    def train(self, show_output: bool = True, data_split=False):
        
        self.save_path = self.dataMaster.split_data_pipe(data_split=data_split)
        self.get_yaml()
        self.load_model()

        print("Beginning training...")
        self.model.train()


    def k_fold_train(self, k: int = 5, show_output: bool = True):
        print(f"Beginning {k}-fold training...")
        self.model.kfold_train(k=k, verbose=show_output)

    def tune(self):
        self.model.tune(data=self.yaml, epochs=30, iterations=300, optimizer='AdamW', plots=False, save=False, val=False)