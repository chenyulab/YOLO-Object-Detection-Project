
#path constants
yolovm_path="C:\\Users\\multimaster\\documents\\Github\\YOLO-Object-Detection-Project\\.yolo\\Scripts\\activate"
exp_utils="C:\\Users\\multimaster\\documents\\Github\\YOLO-Object-Detection-Project\\exp_utils.py"
source "$yolovm_path"

## PREDICTION MODES ## 

# This script has two prediction modes: singular subject and list of subjects
#   to predict just one subject set the many argument to false and set subID to the target subject
#   to predict multiple subjects set the many argument to true and provide the path to a .txt file with a the subjects
#   an example can be found under metadata/distributed_processing/comp1.txt
#                        
expID=310
many=true
CHILD=1
PARENT=0

subID="__20260331_10133"
# subID="__20260324_10133"
subList="Z:\James\YOLO310_JW\310_JW_6_30_Augmented\subPredictionList.txt"

## PREDICTION ARGUMENTS  ##

# There are two scripts used for face processing. 
# one predicts and or removes and normalizes the predictions, to use this script set the post argument to true
# The other smooths the predictions and creates a video with the predictions, to use this set the 
# pred flag to true
# to use both set both flags to true, pred will be exectuted before post 
pre=1
post=1

# The actions below can all be performed during one run, so for a new subject all the flags except 
# RM will be true. The actions are performed in the following order RM -> PRED -> NORM -> SMOOTH -> VIDEO 
# In case a subjects face predictions need be re run one must remove the existing predictions
# this can be done by setting RM to 1
RM=1
# To create face predictions set the PRED argument to 1
PRED=1
# To smooth predictions set SMOOTH to 1
SMOOTH=1
# To create a video with predictions set VIDEO to 1
VIDEO=1

# The actions above can all be performed during one run, so for a new subject all the flags except 
# RM will be true. The actions are performed in the following order RM -> PRED -> NORM -> SMOOTH -> VIDEO 

if ($many); then
    dos2unix "$subList"

    while read line;  do
        python "$exp_utils" "$expID" "$line" --CHILD "$CHILD" --PARENT "$PARENT" --RM "$RM" --PRED "$PRED" --SMOOTH "$SMOOTH" --VIDEO "$VIDEO" --pre "$pre" --post "$post"
        #FOR DEGUGGING echo "$line" >> metadata/final_list.txt
    done < "$subList"
else
    python "$exp_utils" "$expID" "$subID" --CHILD "$CHILD" --PARENT "$PARENT" --RM "$RM" --PRED "$PRED" --SMOOTH "$SMOOTH" --VIDEO "$VIDEO" --pre "$pre" --post "$post" -v 1
    #FOR DEGUGGING echo "$subID" >> metadata/final_list.txt
fi

deactivate

## fyi we call the script twice because they have to be exectued
# in different environments, utralitytics can be installed and executed in
# tf vm(python3.9) but obj_detector uses some python features not in 3.9
# so technically if someone wants to refactor that code we only have to 
# call the script once 