from obj_detector.predictor import PredictorModel  # Importing the PredictorModel class from your predictor module
from pathlib import Path

'''
Created by Jacob Rivera
Spring 2024

Last edit: 03/28/2024

Description:
    Predict objects in input video

    There are many flags that can be passed into the predict_video() func,
    all of them besides the input_video have defaults.

    Set the flags to achieve the desired output.

    
    model_path:
        path to .pt file to predict with

    input_video:
        path to video to predict objects in

    output_video:
        if saving the yolo drawn video (yolo puts the bounding boxes for us)
        then provide a name for the output video, otherwise the output name
            will be based off the input video name

'''

EXP = "351_"
MODEL_NAME = "_E5_"

model_path = Path(r"c:\Users\multimaster\Documents\dark_objects_project\models\E5\weights\best.pt")
target_dir = Path(r"M:\experiment_351\included\__20240504_10090\cam07_video_r\20240504_10090_cam07.mp4")

output_path = Path(r"C:\Users\multimaster\Documents\dark_objects_data\predictions")


input_video = target_dir 
vid = target_dir.parts[-1]
output_video =  output_path / (EXP + "pred" + MODEL_NAME + vid)

def main(target_dir):
    # format prediction output
    input_video = target_dir 
    vid = target_dir.parts[-1]
    output_video =  output_path / (EXP + "pred" + MODEL_NAME + vid)


    # instatiate PredictorModel class with model path
    predictor = PredictorModel(model_path)

    # predict objects in input video,
    predictor.predict_video(
        input_vid=input_video, 
        output_vid_name=None,
        output_dir=output_video, 
        save_annot=False, 
        save_frames=False, 
        save_yolo_vid=True, 
        save_drawn_frames=False, 
        normalize_annot=True, 
        save_conf=True
    )

    

if __name__ == "__main__":
    main(target_dir)