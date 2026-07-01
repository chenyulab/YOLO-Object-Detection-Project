from obj_detector.predictor import PredictorModel  # Importing the PredictorModel class from your predictor module
from obj_detector.data import DataMaster
from pathlib import Path
import os


'''
    ~~~ TO RUN ~~~
    in a terminal or command prompt, run the following commands from the top level of this project

    ! MAC/LINUX
    $ source venv/bin/activate

    ! WINDOWS   
    $ source venv\\Scripts\\activate

    python examples/predict_frames_exp_lvl.py

    deactivate
    ~~~~~~~~~~~~~~
'''
MODEL_PATH = Path("data\\trained_models\\all_data_2_14_mirrored_v8m\\weights\\best.pt")
TOP_EXP_DIR = Path("M:\\experiment_351\\included")

CHILD_FRAMES_DIR_NAME = "cam07_frames_p"
PARENT_FRAMES_DIR_NAME = "cam08_frames_p"



smooth_annotations = True
draw_frames = True
save_individual_drawn_frames = False


def main():

    subject_directories = list(TOP_EXP_DIR.glob("__*"))
    predictor = PredictorModel(MODEL_PATH)

    # instantiate datamaster class
    dataHandler = DataMaster(MODEL_PATH)

    for sub in subject_directories:
        OUTPUT_ROOT = TOP_EXP_DIR.joinpath(sub, 'supporting_files')

        agent = "child"
        output_video_path = OUTPUT_ROOT.joinpath(f"bbox_video_{agent}")
        output_annot_path = OUTPUT_ROOT.joinpath(f"bbox_annotations_{agent}")

        os.makedirs(output_video_path, exist_ok=True)
        os.makedirs(output_annot_path, exist_ok=True)


        # cam 7
        frames_path = TOP_EXP_DIR.joinpath(sub, CHILD_FRAMES_DIR_NAME)

        print(f"Processing: {frames_path}")
        predictor.predict_frames(frames_dir=frames_path, annot_output_path=output_annot_path, drawn_frame_output_path=output_video_path)

        # smooth the annotations
        if smooth_annotations:
            dataHandler.smooth_annotations(input_dir=output_annot_path)
        
        # draw the annotations onto the frames
        if draw_frames:
            dataHandler.batch_draw_bb(images_dir=frames_path, labels_dir=output_annot_path, output_dir=output_video_path, kid_ID=kid_ID,save_frames=save_individual_drawn_frames)
    

        # cam 8
        agent = "parent"
        output_video_path = OUTPUT_ROOT.joinpath(f"bbox_video_{agent}")
        output_annot_path = OUTPUT_ROOT.joinpath(f"bbox_annotations_{agent}")

        os.makedirs(output_video_path, exist_ok=True)
        os.makedirs(output_annot_path, exist_ok=True)

        frames_path = TOP_EXP_DIR.joinpath(sub, "cam08_frames_p")

        print(f"Processing: {frames_path}")
        predictor.predict_frames(frames_dir=frames_path, annot_output_path=output_annot_path, drawn_frame_output_path=output_video_path)

        # smooth the annotations
        if smooth_annotations:
            dataHandler.smooth_annotations(input_dir=output_annot_path)
        
        # draw the annotations onto the frames
        if draw_frames:
            dataHandler.batch_draw_bb(images_dir=frames_path, labels_dir=output_annot_path, output_dir=output_video_path, kid_ID=kid_ID,save_frames=save_individual_drawn_frames)
    

if __name__ == "__main__":
    main()