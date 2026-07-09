import os
from pathlib import Path
import datetime
import argparse
from obj_detector.predictor import PredictorModel  # Importing the PredictorModel class from your predictor module
from obj_detector.data import DataMaster
from tqdm import tqdm 


parser = argparse.ArgumentParser()
parser.add_argument("expID", type=int)
parser.add_argument("subID", type=str)
parser.add_argument("--CHILD", nargs='?', const=0, type=int)
parser.add_argument("--PARENT", nargs='?', const=0, type=int)
# pred args
parser.add_argument("--RM", nargs='?', const=0, type=int)
parser.add_argument("--PRED", nargs='?', const=0, type=int)
# post args 
parser.add_argument("--SMOOTH", nargs='?', const=0, type=int)
parser.add_argument("--VIDEO", nargs='?', const=0,type=int)
# thing about this is that post has to run on a
# different vm than pred so the script will still be have 
# to be called twice for one subject 
parser.add_argument("--pre", nargs='?', const=0, type=int)
parser.add_argument("--post", nargs='?', const=0,type=int)
parser.add_argument("-v", nargs='?', const=0,type=int)

args = parser.parse_args()

model_path = r"Z:\James\FullCVAT_Videos\auto.pt"
predictor = PredictorModel(model_path)
datamaster = DataMaster(model_path)


def save_log(expID, subID, cat_val):
     with open (r"C:\Users\multimaster\documents\Github\YOLO-Object-Detection-Project\metadata\prediction-log.txt", "a") as file:
            file.write(f"{expID} {subID} {datetime.datetime.now()} {cat_val}\n")


def predict_dyad(expID, subID, INPATH, CHILD, PARENT, PRED, RM, SMOOTH, VIDEO, pre, post):
    REL_INPATH = INPATH / subID 
    
    print(f'Loading subject: {subID}')
    if PARENT:
        result = predict_subject(expID, subID, REL_INPATH, "parent", PRED, RM, SMOOTH, VIDEO, pre, post)
     
    if CHILD:
        result =predict_subject(expID, subID, REL_INPATH, "child", PRED, RM, SMOOTH, VIDEO, pre, post)

    if result:
        print(f"SUCCESS {subID} {datetime.datetime.now()}\n")
   

def predict_subject(expID, subID, REL_INPATH, agent, PRED, RM, SMOOTH, VIDEO, pre, post):
    if agent == "child":
        INPATH_FM = REL_INPATH / "cam07_frames_p"
    elif agent == "parent":
        INPATH_FM = REL_INPATH / "cam08_frames_p"

    if not os.path.exists(INPATH_FM):
        print("subject has no FOV frame folder")
        return False
    
    OUTPATH = REL_INPATH / "supporting_files" / f"bbox_video_{agent}"
    AN_PATH = REL_INPATH / "supporting_files" / f"bbox_annotations_{agent}"

    os.makedirs(OUTPATH, exist_ok=True)
    os.makedirs(AN_PATH, exist_ok=True)

    if pre:
        proccess_subject(expID, subID, INPATH_FM, OUTPATH, AN_PATH, PRED, RM, SMOOTH, VIDEO)

    if post:
        postProcess_subject(expID, subID, INPATH_FM, OUTPATH, AN_PATH, SMOOTH, VIDEO)


def proccess_subject(expID, subID, INPATH_FM, OUTPATH, AN_PATH, PRED, RM, SMOOTH, VIDEO): 
    if RM:
        try:
            remove_dir(AN_PATH)
            save_log(expID, subID, "RM")
        except Exception as e:
            save_log(expID, subID, e)
    if PRED:
        try:
            predictor.predict_frames(frames_dir=INPATH_FM, annot_output_path=AN_PATH, drawn_frame_output_path=OUTPATH)
            save_log(expID, subID, "PRED")
        except Exception as e:
            save_log(expID, subID, e)
    
    if RM == 0 & PRED == 0 or RM == 1:
        print("Subject has necesarry data to predict pass PRED flag\n")
        
    return True

def postProcess_subject(expID, subID, INPATH_FM, OUTPATH, AN_PATH, SMOOTH, VIDEO):
    # import here because obj_detector dependencies will break tf conda 
    # have to run it with obj_detector vm 

    if not os.path.exists(AN_PATH):
        print("subject has no annotations frame folder")
        return 

    if SMOOTH:
        try:
            datamaster.smooth_annotations(AN_PATH, 5)
            save_log(expID, subID, "SMOOTH")
        except Exception as e:
            save_log(expID, subID, e)
        
    if VIDEO:
        try:
            datamaster.batch_draw_bb(INPATH_FM, AN_PATH, OUTPATH, subID)
            save_log(expID, subID, "VIDEO")
        except Exception as e:
            save_log(expID, subID, e)

    if VIDEO == 0 & SMOOTH == 0:
        print(f"Subject has necessary to be postprocessed pass VIDEO or SMOOTH flag\n")

def remove_dir(INDIR):
    files = os.listdir(INDIR)

    for f in tqdm(files, desc="deleting files"):
        relative_frame = INDIR / f
        os.remove(relative_frame)

def main():
    IN_PATH = Path(f"M:\\experiment_{args.expID}\\included")
    #IN_PATH = Path(r"C:\Users\multimaster\Documents\cropped_images")
    print(f'Args: expID={args.expID}, subID={args.subID}, CHILD={args.CHILD}, PARENT={args.PARENT}, RM={args.RM}, PRED={args.PRED}, SMOOTH={args.SMOOTH}, VIDEO={args.VIDEO}')
   
    if args.v:
        user_input = input("Are the arguments above correct? Y/N: ")
        user_input = user_input.strip()

        if user_input == "Y":
            predict_dyad(args.expID, args.subID, IN_PATH, args.CHILD, args.PARENT, args.PRED, args.RM, args.SMOOTH, args.VIDEO, args.pre, args.post)
        else:
            print("Script cancelled")
    else:
        predict_dyad(args.expID, args.subID, IN_PATH, args.CHILD, args.PARENT, args.PRED, args.RM, args.SMOOTH, args.VIDEO, args.pre, args.post)

if __name__ == "__main__":
    main()

    