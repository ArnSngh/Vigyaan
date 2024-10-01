import cv2
import torch
from ultralytics import YOLO
from torchvision import models, transforms
import numpy as np

# Load the YOLOv8 model (choose a pre-trained model)
try:
    yolo_model = YOLO('yolov8n.pt')  # Ensure this file is available
except FileNotFoundError:
    print("Error: YOLOv8 model file not found. Please check the path.")
    exit()

# Load a pre-trained ResNet model for classification (Optional)
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()

# Image transformation for ResNet
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Labels for detection (can be expanded based on the scenario)
illegal_objects = ['vehicle', 'person', 'chainsaw', 'gun','scissors', 'bottles', 'book', 'knife']  # Add labels as needed

# Video capture (use the provided path)
video_path = r"C:\Users\Aryan Singh\Desktop\VIgyaan Projects\Forest\footages\footage2.mp4"
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print(f"Error: Could not open video source at {video_path}")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame or end of video reached.")
        break

    # YOLOv8 object detection
    results = yolo_model(frame)

    # Loop through detected objects
    for result in results:
        boxes = result.boxes  # Access detected boxes
        for box in boxes:
            # Extract the bounding box and class label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = yolo_model.names[class_id]

            # Check if the detected object is considered illegal
            if label in illegal_objects:
                # Draw a bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Extract region of interest for ResNet classification (optional)
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    input_tensor = preprocess(roi).unsqueeze(0)

                    # Use ResNet to verify object (optional)
                    with torch.no_grad():
                        output = resnet_model(input_tensor)
                        # Implement further processing with ResNet output if needed

    # Display the frame
    cv2.imshow('Forest Monitoring', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()


#before running the code install neccesaary packages
# pip install ultralytics torch opencv-python numpy
