
import os
from obj_detector.predictor import PredictorModel  # Importing the PredictorModel class from your predictor module
from pathlib import Path

'''
Created by Jacob Rivera
Spring 2024

Last edit: 03/28/2024

Description:
    Predict objects in input image

    There are many flags that can be passed into the predict_image() func,
    all of them besides the img have defaults.

    Set the flags to achieve the desired output.

    model_path:
        path to .pt file to predict with

    img_path:
        path to image to predict objects in

    drawn_frame_output_dir:
        if saving the yolo drawn image (yolo puts the bounding boxes for us)
        then provide a path to a directory for the output image, otherwise the output name
            will be saved in the same location of the input img

    annot_output_dir:
        output directory to place annotation text file containing prediction bounding box info
        if no directory is provided, then the .txt file will be placed in the same dir
            as the input imgage

            
    ~~~ TO RUN ~~~
    ensure virtual enviroment is set in IDE and hit run
    OR
    in a terminal or command prompt, run the following commands from the top level of this project

    ! MAC/LINUX
    $ source venv/bin/activate

    ! WINDOWS   
    $ source venv\\Scripts\\activate

    python examples/predict_image.py

    deactivate
    ~~~~~~~~~~~~~~
'''
# model_path = Path("C:\\Users\\multimaster\\Documents\\GitHub\\YOLO-Object-Detection-Project\\YOLO-Object-Detection-Project\\yolov8m.pt")
model_path = Path("data\\trained_models\\all_data_2_14_mirrored_v8m\weights\\best.pt")
img_path = Path("M:\\experiment_12\\included\\__20151217_16963\\cam07_frames_p\\img_1.jpg") # will be different
drawn_frame_output_dir = Path('M:\\experiment_351\\included\\__20151217_16963\\supporting_files\\bbox_frames_child')
annot_output_dir = Path('M:\\experiment_351\\included\\__20151217_16963\\supporting_files\\bbox_annotations_child')


# drawn_frame_output_dir = img_path.parent
# annot_output_dir = img_path.parent

def main():
    # instatiate PredictorModel class with model path
    predictor = PredictorModel(model_path)

    os.makedirs(drawn_frame_output_dir, exist_ok=True)
    os.makedirs(annot_output_dir, exist_ok=True)


    # predict objects in image
    predictor.predict_image(
        img=img_path, 
        drawn_frame_output_path=drawn_frame_output_dir, 
        annot_output_path=annot_output_dir,
        save_yolo_img=True, 
        save_conf=True, 
        normalize_annot=True
    )



if __name__ == "__main__":
    main()