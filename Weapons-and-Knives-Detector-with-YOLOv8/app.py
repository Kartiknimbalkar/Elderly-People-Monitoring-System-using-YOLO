import cv2
from ultralytics import YOLO

def detect_objects_in_webcam():
    # Load the YOLO model
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    
    # Access the webcam (0 is typically the default camera)
    video_capture = cv2.VideoCapture(0)

    # Check if the webcam opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture a frame from the webcam
        ret, frame = video_capture.read()
        
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Perform object detection on the frame
        results = yolo_model(frame)

        for result in results:
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

        # Display the frame with detections in a window
        cv2.imshow("Webcam Detection", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close any OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

# Call the function to start real-time object detection
detect_objects_in_webcam()
