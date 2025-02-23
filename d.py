import cv2
import torch
import sys
import os

# Add the yolov5 folder to the system path so you can import it
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))

# Import YOLOv5 from your local directory
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.general import scale_boxes  # Import the correct scale_boxes function

# Load the YOLOv5 model from your local directory
model = DetectMultiBackend('yolov5s.pt', device='cpu')  # Make sure you use the correct path to the weights
model.eval()

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB and resize to 640x640 (input size for YOLOv5)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    img_resized = cv2.resize(img, (640, 640))  # Resize to 640x640 for YOLOv5
    
    # Convert to tensor, normalize, and add batch dimension
    img_tensor = torch.from_numpy(img_resized).float()  # Convert to float tensor
    img_tensor /= 255.0  # Normalize the image to [0, 1]
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # Convert HWC to CHW and add batch dimension
    
    # Run object detection
    with torch.no_grad():
        pred = model(img_tensor)  # Run inference

    # Apply non-maxima suppression
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, agnostic=False)

    # Render results on the frame (bounding boxes, labels, and confidences)
    for det in pred[0]:
        if det is not None and len(det):
            # Debugging: Print the shape and contents of 'det' to inspect it
            print("det shape:", det.shape)
            print("det contents:", det)

            # If 'det' is a 1D tensor, reshape it to make it 2D
            if len(det.shape) == 1:
                det = det.unsqueeze(0)  # Reshape to 2D tensor

            # Ensure det is 2D before indexing
            if det.dim() == 2:  # Only index if it's 2D
                # Rescale the coordinates to the original frame size
                det[:, :-1] = scale_boxes(img_tensor.shape[2:], det[:, :-1], frame.shape).round()  # Corrected here

                # Iterate over the detected boxes and draw them
                for *xyxy, conf, cls in det:
                    # Convert the coordinates from float to int
                    x1, y1, x2, y2 = map(int, xyxy)

                    # Draw bounding box using cv2.rectangle
                    color = (255, 0, 0)  # RGB color for the box (red in this case)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                    # Optionally, add label and confidence
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, label, (x1, y1 - 10), font, 0.6, color, 2)

    # Display the frame with object detection
    cv2.imshow('Object Detection - YOLOv5', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
