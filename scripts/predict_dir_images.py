from obj_detector.predictor import PredictorModel  # Importing the PredictorModel class from your predictor module
from obj_detector.data import DataMaster
from pathlib import Path
import os

'''
Created by Jacob Rivera
Spring 2024

Last edit: 03/28/2024

Description:
    Predict objects in a directory of images (or frames from a video)

    There are many flags that can be passed into the predict_frames() func,
    all of them besides the img have defaults.

    Set the flags to achieve the desired output.

    model_path:
        path to .pt file to predict with

    frames_path:
        path to directory containing all images/frames to predict objects in

    drawn_frame_output_dir:
        if saving the yolo drawn image (yolo puts the bounding boxes for us)
        then provide a path to a directory for the output image, otherwise the output name
            will be saved in the same location of the input img

    annot_output_dir:
        output directory to place annotation text file containing prediction bounding box info
        if no directory is provided, then the .txt file will be placed in the same dir
            as the input imgage

    ~~~ TO RUN ~~~
    in a terminal or command prompt, run the following commands from the top level of this project

    ! MAC/LINUX
    $ source venv/bin/activate

    ! WINDOWS   
    $ source venv\\Scripts\\activate

    python examples/predict_frames.py

    deactivate
    ~~~~~~~~~~~~~~
'''
model_path = Path(r"C:\Users\multimaster\Documents\dark_objects_project\models\E7\weights\best.pt")
frames_path = Path("M:\\experiment_351\\included\\__20240718_10115\\cam07_frames_p") # will be different

agent = "child"
output_root =  Path(r"C:\Users\multimaster\Documents\dark_objects_project\predictions\__20240718_10115_fork")
output_video_path = output_root.joinpath(f"bbox_video_{agent}")
output_annot_path = output_root.joinpath(f"bbox_annotations_{agent}")


# output_video_name = "stitched.mp4"
smooth_annotations = True
draw_frames = True
save_individual_drawn_frames = False


def main():
    predictor = PredictorModel(model_path)

    # instantiate datamaster class
    dataHandler = DataMaster(model_path)

    os.makedirs(output_video_path, exist_ok=True)
    os.makedirs(output_annot_path, exist_ok=True)

    kid_ID = frames_path.parts[-2]

    predictor.predict_frames(frames_dir=frames_path, annot_output_path=output_annot_path, drawn_frame_output_path=output_video_path)

    # smooth the annotations
    if smooth_annotations:
        dataHandler.smooth_annotations(input_dir=output_annot_path)
    
    # draw the annotations onto the frames
    if draw_frames:
        dataHandler.batch_draw_bb(images_dir=frames_path, labels_dir=output_annot_path, output_dir=output_video_path, kid_ID=kid_ID,save_frames=save_individual_drawn_frames)
   



if __name__ == "__main__":
    main()