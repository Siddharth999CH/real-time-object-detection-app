import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (you can use 'yolov8n.pt' for faster performance on low-end systems)
model = YOLO("yolov8s.pt")

# Start video capture (webcam)
cap = cv2.VideoCapture(0)

while True: 
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)

    # Plot the results (draw boxes and labels)
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("Real-Time Object Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
