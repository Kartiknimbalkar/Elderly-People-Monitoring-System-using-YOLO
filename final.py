import cv2
import cvzone
import math
from ultralytics import YOLO

def detect_objects_in_webcam():
    # Load YOLO models
    
    yolo_model_fall = YOLO("./Weapons-and-Knives-Detector-with-YOLOv8/runs/detect/Normal_Compressed/weights/best.pt")  # For knife/gun detection
    yolo_model_fall_person = YOLO('./yolov8s.pt')  # For fall detection (using a basic YOLOv8 model)
    
    # Access the webcam (0 is typically the default camera)
    video_capture = cv2.VideoCapture(0)

    # Check if the webcam opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    # Read the classes for the second model
    classnames = []
    with open('classes.txt', 'r') as f:
        classnames = f.read().splitlines()

    while True:
        # Capture a frame from the webcam
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to grab frame.")
            break

        # Perform object detection for knife/gun detection
        results_knives_guns = yolo_model_fall(frame)

        for result in results_knives_guns:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.5:  # Only show detections with confidence > 0.5
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                    color = (0, int(cls[pos]), 255)  # Random color based on class
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Perform object detection for fall detection (person detection)
        results_person = yolo_model_fall_person(frame)

        for info in results_person:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = box.conf[0]
                class_detect = box.cls[0]
                class_detect = int(class_detect)
                class_detect = classnames[class_detect]
                conf = math.ceil(confidence * 100)

                # Implement fall detection using the coordinates x1, y1, x2
                height = y2 - y1
                width = x2 - x1
                threshold = height - width

                if conf > 70 and class_detect == 'person':
                    cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                    cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

                if threshold < 0.5:
                    cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2, colorR=(0, 0, 255))
                else:
                    cvzone.putTextRect(frame, 'Standing', [x1, y1 - 40], thickness=2, scale=1, colorR=(0, 255, 0))

        # Display the frame with detections in a window
        cv2.imshow("Webcam Detection - Fall & Knife/Gun Detection", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close any OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

# Call the function to start real-time object detection
detect_objects_in_webcam()
