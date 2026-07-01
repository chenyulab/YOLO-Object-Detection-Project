import cv2
from ultralytics import YOLO
import time




def main():
    word = ""
    no_no_words = ['person', 'chair']
    CAPITALIZE = False
    count = 0

    # Load the YOLOv8 model
    # model = YOLO('runs/detect/train11/weights/best.pt')
    model = YOLO('model_found.pt')

    # Open the video file
    # video_path = "path/to/your/video/file.mp4"
    cap = cv2.VideoCapture(0)

    # Loop through the video frames
    while cap.isOpened():

        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

        # q = cv2.waitKey(1) & 0xFF

        # Exit when 'q' is pressed
        # if  cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        key = cv2.waitKey(1)  # Wait for a key event for 1 millisecond
        if key == ord('q'):  # Check if the pressed key is 'q'
            cv2.destroyAllWindows()  # Close all OpenCV windows
            break
        # time.sleep(1.5)

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()