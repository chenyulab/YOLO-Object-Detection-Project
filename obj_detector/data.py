import os
import random
import re
import shutil
from pathlib import Path
import datetime
import numpy as np
from tqdm import tqdm
import yaml
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter
import PIL
import cv2

from . import constants


class DataMaster():
    """
    Class to handle all(most) data related operations. 

    ...

    Attributes
    ----------
    dataset_path : str|Path
        a path-like string or Path object
        Points to top level dataset directory with images/ and labels/ subdirs
    class_dict : dict
        dictionary with all classes to be expected in the dataset
        defaultly set to the constant in constants.py, but a unique dict can be passed for processing
    save_path : str|Path
        a path-like string or Path object pointing to location where any output data should be placed
        default: subdir in dataset_path/{today's date}_Training-data/

    Methods
    -------
    split_data_pipe(split:int) -> str|Path
        Divides self.dataset_path into a train and val split.

    list_files_in_directory(self, directory_path, ending:str=None) -> list[str]
        Returns a list all files with a certain ending in the dir directory_path in lexographical order

    generate_mirror_vars(self, put_back:bool = True) -> None
        Generates image and text image variations by mirroring across the x, y, and xy axis, grayscaling, and light blurring
    
    mirror_image_x(self, input_image_path, output_image_path) -> None
        Mirrors and saves an image across the x axis

    mirror_image_y(input_image_path, output_image_path) -> None
        Mirros and saves an image across the y axis

    mirror_image_xy(input_image_path, output_image_path) -> None
        Mirros and saves an image across the x and y axis

    grayscale_image(input_image_path, output_image_path) -> None
        Converts an image to grayscale and saves it

    blur_image(input_image_path, output_image_path) -> None
        Applies a light blur to an image and saves it

    mirror_bounding_box_x_axis(bb_array) -> list[float]
        Returns an array containing bounding box information mirrored over x axis
    
    mirror_bounding_box_y_axis(bb_array) -> list[float]
        Returns an array containing bounding box information mirrored over y axis

    mirror_bounding_box_xy(bb_array) -> list[float]
        Returns an array containing bounding box information mirrored over x and y axis

    process_text_file(self, input_file, output_file, direction)

    batch_gen_blur_levels(self, input_dir, output_dir, threshold)

    get_blur_level(image_path)

    get_obj_blur_levels(self, parent_dir, num_obj, threshold:int=constants.DEFUALT_BLUR_THRESHOLD, output_dir='')

    draw_single_frame(frame, labels_file, drawn_frame, save_drawn_frame=False):





    """
    def __init__(self, dataset_path, class_dict=constants.CLASSES_DICT, color_dict=constants.RGB_DICT, save_path=None) -> None:
        self.dataset_path = Path(dataset_path)
        self.class_dict = class_dict
        self.color_dict = color_dict
        self.save_path = save_path if save_path is not None else Path(self.dataset_path / f'{datetime.date.today().isoformat()}_Training-data')
        self.yaml = None
        # Set a seed for reproducibility
        random.seed(constants.SEED) 


    def split_data_pipe(self, data_split=False, split=constants.DEFUALT_DATA_SPLIT):

        labels = sorted(self.dataset_path.rglob("*labels/*.txt")) # all data in 'labels'

        # Initialize an empty list to store image file paths
        images = []
            # Loop through supported extensions and gather image files
        for ext in constants.SUPPORTED_EXTENSIONS:
            images.extend(sorted((self.dataset_path / 'images').rglob(f"*{ext}")))

        ## possible bug self.save_path is always unique so it is was create before It will create a new instance
        ## which is nested inside the first created instance
        # save_path = Path(self.dataset_path / f'{datetime.date.today().isoformat()}_Split-data')
        self.save_path.mkdir(parents=True, exist_ok=True)

        (self.save_path / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (self.save_path / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        (self.save_path / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (self.save_path / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

        # Create dataset YAML files
        dataset_yaml = self.save_path / 'dataset.yaml'
        with open(dataset_yaml, 'w') as ds_y:
            yaml.safe_dump({
                'path': self.save_path.as_posix(),
                'train': 'train',
                'val': 'val',
                'names': self.class_dict
            }, ds_y)

        self.yaml = dataset_yaml

        # Randomly shuffle the file names using the seeded random function
        random.shuffle(images)
        # Calculate the split point based on an split/1-split ratio ex: 80/20
        split_point = int(split * len(images))
        
        if data_split:
            return self.save_path
        # copy train split
        for img in tqdm(images[:split_point], desc="Copying images"):
            img_path = self.dataset_path / 'images' / img.name
            label_path = self.dataset_path / 'labels' / (Path(img.name).stem + ".txt")

            if not label_path.exists():
                print(f"[SKIP - TRAIN] Missing label: {img.name}")
                continue
            try:
                shutil.copy(img_path, self.save_path / 'train' / 'images' / img.name)
                shutil.copy(label_path, self.save_path / 'train' / 'labels' / label_path.name)
            except Exception as e:
                print(f"[ERROR - TRAIN] {img.name}: {e}")
                continue

        # copy test split
        for img in tqdm(images[split_point:], desc="Copying images"):
            img_path = self.dataset_path / 'images' / img.name
            label_path = self.dataset_path / 'labels' / (Path(img.name).stem + ".txt")

    
            if not label_path.exists():
                print(f"[SKIP - VAL] Missing label: {img.name}")
                continue

            try:
                shutil.copy(img_path, self.save_path / 'val' / 'images' / img.name)
                shutil.copy(label_path, self.save_path / 'val' / 'labels' / label_path.name)

            except Exception as e:
                print(f"[ERROR - VAL] {img.name}: {e}")
                continue



        return self.save_path


    # Sorting function to handle file names with integers
    # without, sorts like 1, 10, 100, 1000, 1001
    def __natural_sort_key(self, filename):
        # Split the filename into parts of digits and non-digits
        parts = [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', filename)]
        return parts


    # Returns a list of all files with a particular ending from a dir
    def list_files_in_directory(self, directory_path:str|Path, ending:str=None) -> list[str]:
        try:
            # Get all files in the directory
            files = os.listdir(directory_path)

            if ending:
                # Filter the list to include only text files (files with a ".txt" extension)
                files_list = [file for file in files if file.lower().endswith(ending)]
            else:
                files_list = [file for file in files if file.lower()]

            # Sort the list of files alphabetically
            sorted_files = sorted(files_list, key=self.__natural_sort_key)

            for i in range(len(sorted_files)):
                sorted_files[i] = os.path.join(directory_path, sorted_files[i])
            return sorted_files

        except OSError as e:
            print(f"Error: {e}")
            return []


    def generate_mirror_vars(self, put_back:bool = True):

        # if user wants mirrored data to not be put with dataset
        if not put_back:
            save_path = self.save_path / 'augmented'
        else:
            save_path = self.save_path

        os.makedirs(save_path, exist_ok=True)

        save_path_images = save_path / 'images'
        save_path_labels = save_path / 'labels'

        (save_path_images).mkdir(parents=True, exist_ok=True)
        (save_path_labels).mkdir(parents=True, exist_ok=True)

        images = self.list_files_in_directory(self.dataset_path / 'images',)

        for img in tqdm(images, desc="Processing images"):

            img_name = os.path.basename(img)
            text_name = os.path.splitext(img_name)[0] + ".txt"

            lbl = os.path.join(self.dataset_path / 'labels', text_name)

            # Try this
            if not os.path.exists(lbl):
                print(f"[WARNING] Missing label file (copy skipped): {lbl}")
                continue

            # Split the filename and extension
            img_name_root, img_ext = os.path.splitext(img_name)
            lbl_name_root, lbl_ext = os.path.splitext(img_name)


            ############### x ################
            img_name_x = Path(img_name_root + "_mirror_acr_x.jpg")
            lbl_name_x = Path(lbl_name_root + "_mirror_acr_x.txt")

            self.mirror_image_x(img, save_path_images / img_name_x)
            self.process_text_file(lbl, (save_path_labels / lbl_name_x), 'x')


            ############### y ################
            img_name_y = Path(img_name_root + "_mirror_acr_y.jpg")
            lbl_name_y = Path(lbl_name_root + "_mirror_acr_y.txt")

            self.mirror_image_y(img, save_path_images / img_name_y)
            self.process_text_file(lbl, save_path_labels / lbl_name_y, 'y')


            ############### xy ###############
            img_name_xy = Path(img_name_root + "_mirror_acr_xy.jpg")
            lbl_name_xy = Path(lbl_name_root + "_mirror_acr_xy.txt")

            self.mirror_image_xy(img, save_path_images / img_name_xy)
            self.process_text_file(lbl, save_path_labels / lbl_name_xy, 'xy')

            ############### grayscale ########
            img_name_bk = Path(img_name_root + "_black_white.jpg")
            lbl_name_bk = Path(lbl_name_root + "_black_white.txt")

            self.grayscale_image(img, save_path_images / img_name_bk)

            shutil.copy(lbl, save_path_labels / lbl_name_bk)

            ############### blur #############
            img_name_bl = Path(img_name_root + "_blurred.jpg")
            lbl_name_bl = Path(lbl_name_root + "_blurred.txt")

            self.blur_image(img, save_path_images / img_name_bl)
            shutil.copy(lbl, save_path_labels / lbl_name_bl)



    # Mirrors images across the x axis -- vertically    u 
    #                                                  ---
    #                                                   n 
    def mirror_image_x(self, input_image_path, output_image_path) -> None:
        try:
            # Open the image file
            with Image.open(input_image_path) as img:
                # Flip the image vertically
                flipped_img = ImageOps.flip(img)

                # Create the output directory if it does not exist
                output_img_dir = os.path.dirname(output_image_path)

                if not os.path.exists(output_img_dir):
                    os.makedirs(output_img_dir)

                # Save the flipped image
                flipped_img.save(output_image_path)
                # if DB: print(f"Flipped image saved as {output_image_path}")
        except IOError:
            print("Unable to load image")

    # Mirrors an image across the y axis -- vertically  <- | ->
    def mirror_image_y(self, input_image_path, output_image_path) -> None:
        try:
            # Open the image file
            with Image.open(input_image_path) as img:
                # Flip the image horizontally using mirror
                flipped_img = ImageOps.mirror(img)

                # Create the output directory if it does not exist
                output_img_dir = os.path.dirname(output_image_path)

                if not os.path.exists(output_img_dir):
                    os.makedirs(output_img_dir)

                # Save the flipped image
                flipped_img.save(output_image_path)
                # if DB: print(f"Flipped image saved as {output_image_path}")
        except IOError:
            print("Unable to load image")

    # Mirrors an image across the y and x axis 
    def mirror_image_xy(self, input_image_path, output_image_path) -> None:
        try:
            # Open the image file
            with Image.open(input_image_path) as img:
                # Flip the image horizontally using mirror
                # flipped_img = ImageOps.flip(img)
                out = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                out = out.transpose(PIL.Image.FLIP_TOP_BOTTOM)

                # mirrored_img = ImageOps.mirror(flipped_img)
                
                # Save the flipped image
                out.save(output_image_path)
                # if DB: print(f"Flipped and mirrored image saved as {output_image_path}")
        except IOError:
            print("Unable to load image")

    # Converts image to grayscale while preserving a 3-channel output image.
    def grayscale_image(self, input_image_path, output_image_path) -> None:
        try:
            with Image.open(input_image_path) as img:
                grayscale_img = ImageOps.grayscale(img).convert("RGB")

                output_img_dir = os.path.dirname(output_image_path)

                if not os.path.exists(output_img_dir):
                    os.makedirs(output_img_dir)

                grayscale_img.save(output_image_path)
        except IOError:
            print("Unable to load image")

    # Applies a light Gaussian blur while leaving labels unchanged.
    def blur_image(self, input_image_path, output_image_path) -> None:
        try:
            with Image.open(input_image_path) as img:
                blurred_img = img.filter(ImageFilter.GaussianBlur(radius=10))

                output_img_dir = os.path.dirname(output_image_path)

                if not os.path.exists(output_img_dir):
                    os.makedirs(output_img_dir)

                blurred_img.save(output_image_path)
        except IOError:
            print("Unable to load image")

    # Mirror bounding box values array across x axis 
    def mirror_bounding_box_x_axis(self, bb_array) -> list[float]:
        return [bb_array[0], 1 - bb_array[1], bb_array[2], bb_array[3]]
    
    # Mirror bounding box values array across y axis 
    def mirror_bounding_box_y_axis(self, bb_array) -> list[float]:
        return [1 - bb_array[0], bb_array[1], bb_array[2], bb_array[3]]

    # Mirror bounding box values array across x and y axis 
    def mirror_bounding_box_xy(self, bb_array) -> list[float]:
        return [1 - bb_array[0], 1 - bb_array[1], bb_array[2], bb_array[3]]


    # transform each line of annotation text file to its respective transformation
    def process_text_file(self, input_file, output_file, direction):

        if not os.path.exists(input_file):
            print(f"[WARNING] Missing label file: {input_file}")
            return
        # Read content from the input file
        with open(input_file, 'r') as infile:
            lines = infile.readlines()

        # iterate over lines from input text file
        for i in range(len(lines)):
            curr_line = lines[i].rstrip('\n')
            curr_line_arr = curr_line.split(' ')
            curr_line_arr[1:] = [float(x) for x in curr_line_arr[1:]]

            if len(curr_line_arr) < 2: 
                continue
            
            # might be a problem with indexing to avoid the obj number (first element)
            if direction == 'y':
                bb_array = self.mirror_bounding_box_y_axis(curr_line_arr[1:])
            elif direction == 'x':
                bb_array = self.mirror_bounding_box_x_axis(curr_line_arr[1:])
            elif direction == 'xy':
                bb_array = self.mirror_bounding_box_xy(curr_line_arr[1:])
            # recreate string to write to output file
            lines[i] = f"{curr_line_arr[0]} {bb_array[0]} {bb_array[1]} {bb_array[2]} {bb_array[3]}\n"

            # Create the output directory if it does not exist
            output_txt_dir = os.path.dirname(output_file)

            if not os.path.exists(output_txt_dir):
                os.makedirs(output_txt_dir)

            # Write the modified content to the output file
            # print(f"output txt file: {output_file}")
            with open(output_file, 'w') as outfile:
                outfile.writelines(lines)

        return



    #### NOT TESTED ####
    # splits data into multiple folds for cross validation training
    def kfold_data(source_dir, output_dir, k, seed):
        random.seed(seed)

        # Set the source directories
        source_text_dir = source_dir + "\\labels"
        source_image_dir = source_dir + "\\images"

        # Get a list of file names in the text folder (assuming the names match the image files)
        file_names = os.listdir(source_text_dir)

        # Randomly shuffle the file names using the seeded random function
        random.shuffle(file_names)

        # Calculate the split point based on an split/1-split ratio ex: 80/20
        fold_size = len(file_names) // k

        subset_files = []
        for i in range(k):
            if i == k-1:
                subset = file_names[i * fold_size :]
            else:
                subset = file_names[i * fold_size : (i + 1) * fold_size]
            subset_files.append(subset)

        for i in range(k):

            training_files  = subset_files[:i] + subset_files[i+1:] # get all files except fold
            training_files = [element for sublist in training_files for element in sublist] # collapse 2d array into 1d

            testing_files = subset_files[i]

            # Set the destination directories
            out_train_text_dir = output_dir + f"\\{i+1}\\train\\labels"
            out_train_image_dir = output_dir + f"\\{i+1}\\train\\images"
            out_test_text_dir =  output_dir + f"\\{i+1}\\test\\labels"
            out_test_image_dir = output_dir + f"\\{i+1}\\test\\images"

            os.makedirs(out_train_text_dir, exist_ok=True)
            os.makedirs(out_train_image_dir, exist_ok=True)
            os.makedirs(out_test_text_dir, exist_ok=True)
            os.makedirs(out_test_image_dir, exist_ok=True)

            for file_name in training_files:
                    try:
                        shutil.copy2(os.path.join(source_text_dir, file_name), os.path.join(out_train_text_dir, file_name))
                        shutil.copy2(os.path.join(source_image_dir, file_name[:-4]+".jpg"), os.path.join(out_train_image_dir, file_name[:-4]+".jpg"))
                    except:
                        print("Could not copy files: " + file_name)
                        continue
            
            for file_name in testing_files:
                try:
                    shutil.copy2(os.path.join(source_text_dir, file_name), os.path.join(out_test_text_dir, file_name))
                    shutil.copy2(os.path.join(source_image_dir, file_name[:-4]+".jpg"), os.path.join(out_test_image_dir, file_name[:-4]+".jpg"))
                except:
                    print("Could not copy files: " + file_name)
                    continue

    '''
    Generates blur values for each image in a directory
    Outputs a text file containing the blur values and overall blur ratio

    Only set to handle .jpg images. If other image types are used, change
    the following line: 'if image[-4:] != ".jpg": continue' 
    to the appropriate file extension.
    ''' 
    ### NOT TESTED ###
    # !!! need to add safety catches
    def batch_gen_blur_levels(self, input_dir, output_dir, threshold):
        img_count = 0

        # get list of all files in dir
        file_name_list = os.listdir(input_dir)

        # if no new name given for text file, use default and put it in
        if output_dir.split('\\')[-1][-4:] != ".txt":
            out_path = os.path.join(output_dir,"images_blur_values.txt")

        blurry_count = 0

        with open(out_path, 'w') as file:
            for image in file_name_list:
                # if not an image, skip
                if image[-4:] != ".jpg": continue

                blurry_lvl = self.get_blur_level( os.path.join(input_dir, image))

                if blurry_lvl < threshold: blurry_count += 1
                img_count += 1

                formatted_string = '{:<30}'.format(image)  # Adjust width as needed for strings
                formatted_float = '{:<10.4f}'.format(blurry_lvl)  # Adjust width and precision for floats

                # Write the formatted string and float to the file
                file.write(f"{formatted_string} {formatted_float}\n")

        # Read the existing content of the file
        with open(out_path, 'r') as file:
            existing_content = file.read()

        to_add = f"\n\n{blurry_count} blurry images out of {img_count}\nRatio: { blurry_count/ img_count}\n"
        all_content = to_add + existing_content

        # Write the combined content back to the file
        with open(out_path, 'w') as file:
            file.write(all_content)

        print(f"\n\n{blurry_count} blurry images out of {img_count}\nRatio: { blurry_count/ img_count}")

        cv2.destroyAllWindows()

        return
    
    ###
    # get blur levels of object level directories
    #
    # input_dir: directory containing object level directories
    # num_obj: number of object level directories
    # threshold: threshold for blurry level
    # output_dir: directory to put obj blur levels text files,
    #   if none given, will put in src dir for each object
    #
    # output: none, creates text files in output_dir
    ###
    ### NOT TESTED ###
    # !!! need to add safety catches
    def get_obj_blur_levels(self, parent_dir, num_obj, threshold:int=constants.DEFUALT_BLUR_THRESHOLD, output_dir=''):

        for i in range(num_obj):
            img_count = 0

            # input and output for object dir
            input_dir = os.path.join(parent_dir, str(i))
            output_dir = os.path.join(parent_dir, str(i)) if output_dir == '' else output_dir

            # get list of all files in dir
            file_name_list = os.listdir(input_dir)

            # create obj text file to hold blurry vals
            out_path = os.path.join(output_dir, f"obj{i}_image_blur_values.txt")
            blurry_count = 0

            with open(out_path, 'w') as file:
                for image in file_name_list:
                    # if not an image, skip
                    if image[-4:] in constants.SUPPORTED_EXTENSIONS: continue

                    blurry_lvl = self.get_blur_level( os.path.join(input_dir, image))
                    if blurry_lvl < threshold: blurry_count += 1
                    img_count += 1

                    formatted_string = '{:<30}'.format(image)  # Adjust width as needed for strings
                    formatted_float = '{:<10.4f}'.format(blurry_lvl)  # Adjust width and precision for floats

                    # Write the formatted string and float to the file
                    file.write(f"{formatted_string} {formatted_float}\n")

            with open(out_path, 'a') as file:
                file.write("\n\n{} blurry images out of {}\n".format(blurry_count, img_count))
                file.write("Ratio: {}".format( blurry_count/ img_count))
            print("\n\nObject {}  --  {} blurry images out of {}\n".format(i, blurry_count, img_count))
            print("Ratio: {}".format( blurry_count/ img_count))

        cv2.destroyAllWindows()
        return
    

    '''
    Generates and returns blur value for a single image
    '''
    ### NOT TESTED ###
    @staticmethod
    def get_blur_level(image_path):
        img = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(img_gray, cv2.CV_64F).var()

        return variance



    def draw_single_frame(self, frame, labels_file, drawn_frame, save_drawn_frame=False):
        #print(labels_file)
        with open(labels_file, 'r') as file:
            lines = file.readlines()

        img = Image.open(frame)
        draw = ImageDraw.Draw(img)

        img_width, img_height = img.size

        # Calculate the font size based on image size
        base_font_size = 15  # Set your base font size here
        font_size = int(base_font_size * min(img_width, img_height) / 800)  # Adjust the divisor as needed

        base_line_width = 3
        line_width = int(base_line_width * min(img_width, img_height) / 800)  # Adjust the divisor as needed

        base_padding = 3
        padding = int(base_padding * min(img_width, img_height) / 800)  # Adjust the divisor as needed

        # Define font and size
        font = ImageFont.truetype(r"C:\Users\multimaster\Documents\Github\YOLO-Object-Detection-Project\data\fonts\lato\Lato-Bold.ttf", font_size)  # Adjust font and size as needed

        for line in lines:
            # Split the line into components
            components = line.strip().split()
            # image_name = components[0].replace('.txt', '.jpg')
            obj_num = int(components[0])
            bbox_info = list(map(float, components[1:]))

            # Get image width and height

            # Convert bounding box coordinates to YOLO format
            x_center = bbox_info[0] * img_width
            y_center = bbox_info[1] * img_height
            box_width = bbox_info[2] * img_width
            box_height = bbox_info[3] * img_height

            # Calculate box coordinates
            x_min = x_center - (box_width / 2)
            y_min = y_center - (box_height / 2)
            x_max = x_center + (box_width / 2)
            y_max = y_center + (box_height / 2)

            if (y_min > y_max):
                temp = y_min
                y_min = y_max
                y_min = temp
            if (x_min > x_max):
                temp = x_min
                x_min = x_max
                x_min = temp

            color = self.color_dict[obj_num] if obj_num in self.color_dict else 'rgb(236, 3, 252)'


            box_text = f" {self.class_dict[obj_num]} "
            # text_bbox = draw.textbbox((0, 0), box_text, font)

            # Draw bounding box
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=line_width)

            # Draw text
            left, top, right, bottom = draw.textbbox((x_min, y_min - font_size), box_text, font=font)
            draw.rectangle((left, top-line_width, right+line_width, bottom), fill=color)
            draw.text((x_min, y_min-font_size), box_text, font=font, fill="white")


        if save_drawn_frame: img.save(drawn_frame)
        return img



    # draws annotations onto each frame of video
    # expects annotations to be saved as text files, one per frame
    #
    # input_vid: path to input video
    # output_vid: path/name of output video
    # predictions_dir: path to directory containing annotations
    #
    # returns: nothing
    def draw_annot_on_video(self, input_vid, output_vid, predictions_dir):
        cap = cv2.VideoCapture(str(input_vid))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_vid, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))

        predictions_files_arr = self.list_files_in_directory(predictions_dir, '.txt')
        count = 0

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if success:
                drawn_frame = self.draw_single_frame(frame, predictions_files_arr[count], '')
                out.write(drawn_frame)
                count += 1
            # Break the loop if the end of the video is reached
            else:
                break

        # Release the video capture object and close the display window
        cap.release()
        out.release()
        cv2.destroyAllWindows()



    # Given two bounding box annotations and the number of frames to fill in for,
    # returns a list of bounding box coordinates arrays
    # 
    # Averages the annotations across the missed frames
    def get_equidistant_points(self, start, end, num_points, sig_figs=5):
        if num_points < 2:
            raise ValueError("Number of points must be at least 2.")
        # elif start[0] != end[0]:
        #     raise ValueError(f"Bounding box annotations must be for the same object.\nStart: obj {start[0]}\nEnd:   obj {end[0]}")
        
        # Calculate the step size for interpolation
        step_size = 1.0 / (num_points - 1)

        # Perform linear interpolation
        points = [[round(start[0] + i * step_size * (end[0] - start[0]), sig_figs), 
                round(start[1] + i * step_size * (end[1] - start[1]), sig_figs),
                round(start[2] + i * step_size * (end[2] - start[2]), sig_figs),
                round(start[3] + i * step_size * (end[3] - start[3]), sig_figs),
                round(start[4] + i * step_size * (end[4] - start[4]), sig_figs)]

                for i in range(1, num_points - 1)]

        return points





        
    '''
    Smooths bounding box coordinates by extrapolating points given the curr file
        has the obj, and one of the next files has the obj. It will fill in the files
        missing the obj between the curr and the file that it is found in. 

    Will smooth over as many files in the array that there are, so limit the array size of
        next_files to limit how many extrapolated annotations will be added.
    '''
    def check_window(self, curr_file, next_files):
        # num_objs = 26
        curr_dict = self.txt_to_dict(curr_file)
        '''
        current dict has the items that i want to check if future frames are missing.
        '''
        file_dicts = []
        smooth_files = []

        # create an array of dictionarys of the next n files
        for f in next_files:
            out = self.txt_to_dict(f)
            # print(f"out: {out}")
            file_dicts.append(out)

        # loop through all objs in curr_dict
        for i in curr_dict.keys():

            count = 0

            # loop through next n files
            index = 0
            missed = 0

            # loop through forward file dictionaries
            for d in file_dicts:
                count += 1

                # check if obj from curr is in one of the next files,
                # when it is, get the missing points
                if i in d.keys() and missed == 1:
                    avg_annot = self.get_equidistant_points(curr_dict[i], d[i], count + 1, sig_figs=5)
                    # print(f"annotation num: {len(avg_annot)}")
                    # print(f"count : {count}")
                    # print(f"index : {index}")

                    # add extrapolated points to files missing it
                    for fix in range(index):
                        #with open(r"C:\Users\multimaster\documents\retinaFace_dilab\metadata\debugg-lerp.txt", "a") as f:
                            # delete non interpolated file and change its extension
                        curr = next_files[fix]
                        parts = curr[:-4].split("_")

                        if parts[-1] == "lerp":
                            next = curr
                        else:
                            next = curr[:-4] + "_lerp.txt"
                            smooth_files.append(curr)
                            shutil.copyfile(curr, next)
                        self.add_line_to_txt(next, i, avg_annot[fix])
                    missed = 0
                    break
                elif (i not in d.keys() ):
                    missed = 1
                elif i in d.keys() and missed == 0:
                    break
                index += 1
        return smooth_files

    # input: 
    #   input_dir: directory containing text files
    #   max_skips: number of frames to fill in missing objects
    # output:
    #   none
    def smooth_annotations(self, input_dir, max_skip=constants.MAX_SKIPS):

        smooth_files = []
        text_files = self.list_files_in_directory(input_dir, '.txt')
        text_files = [os.path.join(input_dir, j) for j in text_files]
        # print(text_files)

        for i in tqdm(range(0, len(text_files) - max_skip- 2), total=len(text_files) - max_skip, desc="Smoothing annotations..."):
            # gets the next max_skips + 1 files to check for skipping objs
            look_aheads = [text_files[i + j] for j in range(1, max_skip + 2)]

            temp_smooth = self.check_window(text_files[i], look_aheads)
            
            for temp in temp_smooth:
                if temp not in smooth_files:
                    smooth_files.append(temp)

        
        # remove old files that were lerped 
        for rm_file in tqdm(smooth_files, desc="cleaning up files.."):
            ## DEGUGGING ## 
            #temp_path = rm_file.split("\\")
            #temp_path.pop(5)
            #temp_path.insert(5, "temp")
            #mv_path = "\\".join(temp_path)
            os.remove(rm_file)

    # returns a dictionary version of bb txt file with
    # obj num as the key
    def txt_to_dict(self, file_path):

        output_dict = {}
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read the file line by line
            for line in file:
                # Process each line as needed
                line = line.strip()
                line_arr = re.split(r'\s+', line)
                output_dict[int(line_arr[0])] = [float(item) for item in line_arr[1:]]
        return output_dict 



    # given a text file, bounding box annotation line will be added
    def add_line_to_txt(self, file_path, key, arr):
        with open(file_path, 'a') as file:
            file.write(f"{key} {arr[0]} {arr[1]} {arr[2]} {arr[3]} {arr[4]}\n")


    def batch_draw_bb(self, images_dir, labels_dir, output_dir, kid_ID, save_frames=False):

        os.makedirs(output_dir, exist_ok=True)

        all_images = self.list_files_in_directory(images_dir, '.jpg')
        all_annots = self.list_files_in_directory(labels_dir, ".txt")

        # Get the first image to retrieve dimensions
        first_image = cv2.imread(os.path.join(images_dir,all_images[0]))
        height, width, _ = first_image.shape

        output_video_name = output_dir.joinpath(f"{kid_ID}_predicted.mp4")
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID', 'MJPG', etc.
        out = cv2.VideoWriter(str(output_video_name), fourcc, constants.STITCH_VIDEO_FPS, (width, height))
        for i in tqdm(range(len(all_images)), desc="Drawing annotations on images..."):
            img_path = os.path.join(images_dir, all_images[i])

            text_file = os.path.join(labels_dir, all_annots[i])

            path, img_root = os.path.split(all_images[i])
            drawn_img = os.path.join(output_dir, img_root[:-4] + "_drawn.jpg")

            pred_frame = self.draw_single_frame(img_path, text_file, drawn_img, save_frames )
            # Convert the Pillow image to a NumPy array
            pred_frame = np.array(pred_frame)
            # print(numpy_array, np.size(numpy_array))
            if pred_frame.shape[-1] == 4:
                pred_frame = cv2.cvtColor(pred_frame, cv2.COLOR_RGBA2BGRA)
            else:
                pred_frame = cv2.cvtColor(pred_frame, cv2.COLOR_RGB2BGR)
            out.write(pred_frame)
            
        # Release VideoWriter and destroy any OpenCV windows
        out.release()
        cv2.destroyAllWindows()

        print(f"Video saved to: {output_video_name}\n")
