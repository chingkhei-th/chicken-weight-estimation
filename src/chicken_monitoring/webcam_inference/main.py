import torch
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression



MODEL_PATH = r"chicken_monitoring\assets\model\chicken.pt" # update your path and model name
CSV_PATH = r"chicken_monitoring\assets\datasets\area_weight.csv" # Update your path and csv file name

# Load the YOLOv8 model
model = YOLO(MODEL_PATH) 


# Load the linear regression model
def linear_regression():
    data = pd.read_csv(CSV_PATH)
    X = data['Area'].values.reshape(-1, 1)
    y = data['Weight'].values
    LRmodel = LinearRegression()
    LRmodel.fit(X, y)
    return LRmodel



def webcam_inference(LRmodel, model, ):
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame from webcam")
            break

        # Run the YOLOv8 model on the frame
        results = model(frame)

        # Initialize the total weight for each frame
        total_weight = 0.0

        # Process the results
        for r in results:
            boxes = r.boxes
            masks = r.masks.xy if r.masks is not None else []
            labels = r.names

            # Iterate through detected objects
            for i, (box, mask) in enumerate(zip(boxes, masks)):
                label = labels[0]  # Assuming single class 'chicken'
                if label == 'chicken':
                    # Convert mask segments to polygon format
                    polygon = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))

                    # Create an empty binary mask
                    mask_shape = frame.shape[:2]  # Shape of the original image (height, width)
                    mask_binary = np.zeros(mask_shape, dtype=np.uint8)

                    # Fill the binary mask with the polygon
                    cv2.fillPoly(mask_binary, [polygon], 255)

                    # Find contours in the mask
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Calculate the area of the segmented chicken
                    area = sum(cv2.contourArea(contour) for contour in contours)

                    # Predict the weight based on the area
                    predicted_weight = LRmodel.predict([[area]])

                    # Add the predicted weight to the total weight
                    total_weight += predicted_weight[0]

                    # Draw the bounding box and predicted weight on the frame
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"chicken: {predicted_weight[0]:.2f} g", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the total weight in the top-right corner
        cv2.putText(frame, f"Total Weight: {total_weight:.2f} g", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Exit the loop if the 'q' key is pressed or the window is closed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is the ASCII code for the Esc key
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    LRmodel = linear_regression()
    webcam_inference(LRmodel, model)