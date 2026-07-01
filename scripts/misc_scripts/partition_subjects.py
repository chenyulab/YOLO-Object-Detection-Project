import os
import re
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("expID", type=int)
parser.add_argument("partition_num", type=int)
args = parser.parse_args()

# read exp directory and filter out non subject foldres 
def get_subjects(expID):
    pat = r'__\w+_\w+'

    subjects_dir = f"M:\\experiment_{expID}\\included"
    subjects = os.listdir(subjects_dir)
    subjects = [s for s in subjects if re.match(pat, s)]

    return subjects

# divide sub of list into n partions
def partition_subjects(expID, partition_num):
    subjects = get_subjects(expID)

    n = len(subjects) // partition_num  
    onset = 0
    offset = n
 
    for i in range(1, partition_num+1):
        if i == partition_num:
            partition = subjects[onset:]
        else:
            partition = subjects[onset:offset]

        with open (f"C:\\Users\\multimaster\\documents\\YOLO-Object-Detection-Project\\metadata\\partition_{i}.txt", "w+") as file:
            for subject in partition:
                file.write(str(subject)+"\n")

        onset = offset
        offset = onset + n

def main():
    partition_subjects(args.expID, args.partition_num)

if __name__ == "__main__":
    main()