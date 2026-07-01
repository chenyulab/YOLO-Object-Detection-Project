
import cv2
from tqdm import tqdm
### stiches two or four videos together ###
### used to compare object detction ouputs ##

vi1 = cv2.VideoCapture(r"M:\experiment_351\included\__20240504_10090\supporting_files\bbox_video_child\__20240504_10090_predicted.mp4")
vi2 = cv2.VideoCapture(r"C:\Users\multimaster\Documents\dark_objects_project\predictions\__20240504_10090_truck\bbox_video_child\__20240504_10090_predicted.mp4")
#vi3 = cv2.VideoCapture(r"C:\Users\multimaster\Documents\GitHub\YOLO-face_detection\face_detection\val_vids\pred\pred_E5_20231015_10090_cam07.mp4\predicted.mp4")
#vi4 = cv2.VideoCapture(r"C:\Users\multimaster\Documents\GitHub\YOLO-face_detection\face_detection\val_vids\pred\pred_E6_20231015_10090_cam07.mp4\predicted.mp4")

frame_width = int(vi1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vi1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vi1.get(cv2.CAP_PROP_FPS)

total_frame_count = int(vi1.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total = total_frame_count)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out_dir = r"Z:\elton\DI\dark_objects\35353_comp_truck.mp4"
output_video = cv2.VideoWriter(out_dir, fourcc, fps, (frame_width * 2, frame_height * 1))

count = 0

with tqdm(total=total_frame_count, desc="stitching videos..") as pbar:
    while True:
        ret1, frame1 = vi1.read()
        ret2, frame2 = vi2.read()
        #ret3, frame3 = vi3.read()
        #ret4, frame4 = vi4.read()

        if not ret1 or not ret2: #or not ret3 or not ret4:
            break

        h1 = frame1
        h2 = frame2
        #h1 = cv2.vconcat([frame1, frame2])
        #h2 = cv2.vconcat([frame3, frame4])
        full = cv2.hconcat([h1, h2])

        output_video.write(full)
        pbar.update(1)

vi1.release()
vi2.release()
#vi3.release()
#vi4.release()
output_video.release()
cv2.destroyAllWindows()
