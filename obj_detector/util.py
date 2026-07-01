import cv2
import os
import shutil
import obj_detector as op
from tqdm import tqdm
from obj_detector import constants

# import constants


# if number of frames to be over 04d, then change the 4 in the f-string to the number of digits
def extract_all_frames(input_vid, output_dir="", frames_prefix="frame_", debug=0):
    # Open the video file
    video = cv2.VideoCapture(input_vid)

    # Check if the video is opened successfully
    if not video.isOpened():
        print("Error opening video file")
        return
    
    frame_count = 0

    if output_dir == "":
        video_pre_path, video_name = os.path.split(input_vid)
        video_name_prefix = video_name.split('.')[0]
        output_dir = os.path.join(video_pre_path, video_name_prefix + "_frames")
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)


    while True:
        # Read a frame from the video
        ret, frame = video.read()

        # Check if frame is read correctly
        if not ret:
            break

        # Save the frame as an image in the output folder
        frame_filename = f"{frames_prefix}{frame_count:04d}.jpg"  # You can adjust the filename pattern
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        # Display the frame (optional)
        if debug: cv2.imshow('Frame', frame)

        # Increment frame count
        frame_count += 1

        # Break the loop on pressing 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture object and close windows
    video.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames. Saved in {output_dir}")





'''
Reads json config file to create the .yaml file needed for YOLO training
Generates information in .yaml based on json config file

'''
def make_config_json(json_config):
    out_name = json_config["training"]["data"]
    dataset_path = json_config["dataset_folder"]

    yaml_config = open(f'{out_name}','w')
    yaml_config.write(f'path: {dataset_path}\n')
    yaml_config.write('train: train\n')
    yaml_config.write('val: test\n')
    yaml_config.write('names:\n')

    obj_num = json_config["constants"]["NUM_OBJS"]
    for i in range(obj_num):
        yaml_config.write(f'  {i}: {json_config["objects"][str(i)].strip()}\n')

    yaml_config.close()

    return



def make_config(out_name, dataset_path, obj_num, obj_dict):
    # out_name = json_config["training"]["data"]
    # dataset_path = json_config["dataset_folder"]

    yaml_config = open(f'{out_name}','w')
    yaml_config.write(f'path: {dataset_path}\n')
    yaml_config.write('train: train\n')
    yaml_config.write('val: test\n')
    yaml_config.write('names:\n')

    for i in range(obj_num):
        yaml_config.write(f'  {i}: {obj_dict[i].strip()}\n')

    yaml_config.close()

    return



def gather_obj_lvl_data(input_dir, output_dir, num_objs):
    
    # INPUT_DIR = "C:\\Users\\multimaster\\Desktop\\Final_dataset_11_13_23\\11_13_23\\original_data_corrected"
    # OUTPUT_DIR = "C:\\Users\\multimaster\\Desktop\\Final_dataset_11_13_23\\11_13_23\\obj_level_data"
    # NUM_OBJ = 27


    os.makedirs(output_dir, exist_ok=True)

    obj_labels_dict = {}

    # instantiate key for each obj in dict
    for i in range(num_objs):
        obj_labels_dict[str(i)] = []

    # make a dir for each obj
    for i in range(num_objs):
        obj_dir = os.path.join(output_dir, str(i))
        os.makedirs(obj_dir, exist_ok=True)


    source_labels_dir = os.path.join(input_dir, "labels")
    source_images_dir = os.path.join(input_dir, "images")

    # Get a list of file names in the text folder (assuming the names match the image files)
    file_name_list = os.listdir(source_labels_dir)

    ########################
    # def process_labels_file

    file_name = file_name_list[0]

    for file_name in file_name_list:
        print(file_name)
        if file_name[-4:] != ".txt":
            continue
        

        # gete full path to be able to copy files later
        full_file_path = os.path.join(source_labels_dir, file_name)

        # Open the file in read mode
        with open(full_file_path, 'r') as file:
            # Read the file line by line
            for line in file:
                # print(line)
                if line == " " or line=="":
                    continue
                line = line.strip() # remove \n
                line_arr = line.split(' ')

                obj_num = str(line_arr[0])
                
                print(obj_num)
                # add label file and line to obj dict array
                obj_labels_dict[obj_num].append([file_name, line])

                # create path to put image
                out_obj_dir = os.path.join(output_dir, str(obj_num))
                # copy image file over
                shutil.copy2(os.path.join(source_images_dir, file_name[:-4]+".jpg"), os.path.join(out_obj_dir, file_name[:-4]+".jpg"))

                
    for i in range(num_objs):
        obj_dir = os.path.join(output_dir, str(i), 'all_labels.txt')

        obj_dict_entry = obj_labels_dict[str(i)]

        with open(obj_dir, 'w') as file:
            for line in obj_dict_entry:
                file.write(line[0] + " " + line[1] + "\n")




# Function to calculate the image blur level
def calculate_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()



def split_data_by_blur(source_dir, destination_dir, threshold):
    
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    os.makedirs(os.path.join(destination_dir,'images'), exist_ok=True)
    os.makedirs(os.path.join(destination_dir,'labels'), exist_ok=True)


    # Iterate through all jpg files in the source directory
    for filename in os.listdir(os.path.join(source_dir,'images')):
        # print(filename)
        if filename.endswith('.jpg'):
            img_filepath = os.path.join(source_dir, 'images', filename)
            txt_filepath = os.path.join(source_dir, 'labels', filename[:-4] + '.txt')

            # Read the image using OpenCV
            img = cv2.imread(img_filepath)
            
            # Calculate the blur level
            blur_level = calculate_blur(img)
        
            # If the blur level is above the threshold, copy the image to the destination directory
            if blur_level > threshold:
                img_destination_path = os.path.join(destination_dir, 'images', filename)
                shutil.copyfile(img_filepath, img_destination_path)

                label_destination_path = os.path.join(destination_dir, 'labels', filename[:-4] + '.txt')
                shutil.copyfile(txt_filepath, label_destination_path)

                print(f"Copied {filename} - Blur level: {blur_level}")

    print("Done processing images.")




def frames_to_video(input_dir, output_video_path, fps=constants.STITCH_VIDEO_FPS):
    # Get the list of image files in the input directory
    # image_files = [f for f in os.listdir(input_dir) if f.endswith('.png') or f.endswith('.jpg')]
    image_files = op.data_processing.smooth.list_files_in_directory(input_dir, ".jpg")
    if not image_files:
        print("No image files found in the directory.")
        return


    # Get the first image to retrieve dimensions
    first_image = cv2.imread(os.path.join(input_dir,image_files[0]))
    height, width, _ = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID', 'MJPG', etc.
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Write each frame to the video
    # for img in tqdm(os.listdir(input_dir),f"Processing frames... {dir_label}"):

    for image_file in tqdm(image_files, desc="Stitching frames into video..."):
        frame = cv2.imread(os.path.join(input_dir,image_file))
        out.write(frame)

    # Release VideoWriter and destroy any OpenCV windows
    out.release()
    cv2.destroyAllWindows()

    print(f"Video saved to: {output_video_path}")