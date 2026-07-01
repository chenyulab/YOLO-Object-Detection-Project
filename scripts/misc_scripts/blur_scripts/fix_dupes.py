
from pathlib import Path
import os
import re

from tqdm import tqdm

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
model_path = Path("All_Data_Trainings\\all_data_2_14_mirrored_v8m\\weights\\best.pt")
# output_path = Path("Z:\\Jacob\\YOLO_Predicted")

TOP_EXP_DIR = Path("Z:\\Jacob\\YOLO_Predicted")


def get_text_files(directory):
    text_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            text_files.append(os.path.join(directory, filename))
    return text_files
# Sorting function to handle file names with integers
# without, sorts like 1, 10, 100, 1000, 1001
def __natural_sort_key( filename):
    # Split the filename into parts of digits and non-digits
    parts = [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', filename)]
    return parts


# Returns a list of all files with a particular ending from a dir
def list_files_in_directory( directory_path:str|Path, ending:str=None) -> list[str]:
    try:
        # Get all files in the directory
        files = os.listdir(directory_path)

        if ending:
            # Filter the list to include only text files (files with a ".txt" extension)
            files_list = [file for file in files if file.lower().endswith(ending)]
        else:
            files_list = [file for file in files if file.lower()]

        # Sort the list of files alphabetically
        sorted_files = sorted(files_list, key=__natural_sort_key)

        for i in range(len(sorted_files)):
            sorted_files[i] = os.path.join(directory_path, sorted_files[i])
        return sorted_files

    except OSError as e:
        print(f"Error: {e}")
        return []

def main():

    subject_directories = list(TOP_EXP_DIR.glob("__*"))


    for sub in subject_directories:
        sub = Path(sub) / "predicted_labels"
        print(sub)

        files = list_files_in_directory(sub,".txt")
        for file in tqdm(files, desc=f"{sub}"):
            data_dict = {}
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Check if line is not empty
                        parts = line.split(" ")
                        key = int(parts[0])
                        value = float(parts[-1])
                        if key not in data_dict or value > data_dict[key][-1]:
                            data_dict[key] = (line, value)

            with open(file, 'w') as f:
                for key in sorted(data_dict.keys()):
                    f.write(data_dict[key][0] + '\n')



if __name__ == "__main__":
    main()