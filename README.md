# Chicken weight estimation

### Tools used

- **YOLOv8** → For Training model for chicken detection
- **Roboflow** → For dataset and image annotation
- **Scikit-learn** → For training LinearRegression model
- **Opencv-python** → For Image processing
- **PyTorch 2.3.0** - CUDA 12.1 → To leverage the power of GPUs for training the model ( Not required when running using Google Colab )
- **Python version** - 3.11

## Project Setup

Prerequisites before running the Inference program:

- Must have a **model** (`.pt`) file trained on segmented chicken dataset, and
- **CSV** file containing relation between area of the chicken and its weight.
- **Poetry** install in your system
    
    If you don’t have already install Poetry on your system:
    
    [Introduction | Documentation | Poetry - Python dependency management and packaging made easy](https://python-poetry.org/docs/#installation)
    

### Steps

1. Clone the project.
2. Go to the project directory and run: 
    
    ```bash
    poetry install --sync
    ```
    

---

## **Theoretical Background**

**Object Detection and Segmentation**

**Object detection** and **segmentation** are fundamental tasks in computer vision, enabling the localization and delineation of objects of interest within an image or video stream. The **YOLOv8 (You Only Look Once)** model employed in this notebook is a state-of-the-art object detection and instance segmentation algorithm that utilizes a single neural network to perform both tasks simultaneously.

The YOLO family of models, including YOLOv8, is known for its real-time performance and high accuracy, making it well-suited for applications that require efficient and reliable object detection and segmentation. The YOLOv8 model is an evolution of its predecessors, incorporating architectural improvements and advanced training techniques to enhance its performance further.

**Instance Segmentation**

**Instance segmentation** is a computer vision task that combines object detection and semantic segmentation. It not only identifies and localizes individual objects within an image but also delineates their precise boundaries at the pixel level. This capability is particularly valuable in the context of broiler chicken weight estimation, as it allows for accurate measurement of the segmented area, which serves as a crucial feature for weight prediction.

**Linear Regression**

**Linear regression** is a fundamental machine learning technique used for predicting a continuous target variable based on one or more input features. In this notebook, linear regression is employed to estimate the weight of broiler chickens based on their segmented area as the input feature. The underlying assumption is that there exists a linear relationship between the area of a chicken and its weight, which can be modeled using a linear equation.

While linear regression is a simple and interpretable model, it may not capture more complex, non-linear relationships between the input features and the target variable. However, it serves as a baseline approach and provides a foundation for further exploration of more advanced regression techniques, such as non-linear regression models or ensemble methods.

## **Methodology**

**1. Loading the YOLOv8 Model**

The notebook (`train_yolov8.ipynb`) begins by installing the `ultralytics` package, which provides the YOLOv8 model implementation. The model is then loaded from a pre-trained weight file (`best.pt`). Pre-trained models are crucial in computer vision tasks, as they encapsulate the knowledge learned from vast amounts of training data, enabling efficient transfer learning and adaptation to specific domains.

**2. Running Inference (image_inference / webcam_inference)**

The loaded YOLOv8 model is used to perform object detection and instance segmentation on a test images. The `model` function is called with the image path as an argument, and the resulting `results` object contains the detected bounding boxes, class labels, and segmentation masks for the broiler chickens present in the image.

The inference process involves several steps:

a. **Image Preprocessing**: The input image is resized and normalized to match the expected input format of the YOLOv8 model.

b. **Forward Pass**: The preprocessed image is passed through the YOLOv8 model, which consists of a deep neural network architecture designed for efficient object detection and instance segmentation.

c. **Post-processing**: The model's output is post-processed to extract the relevant information, such as bounding boxes, class labels, and segmentation masks, for each detected object.

The YOLOv8 model's ability to perform instance segmentation is particularly valuable in this application, as it provides pixel-level masks for each detected broiler chicken, enabling accurate area measurement.

**3. Extracting Segmentation Masks and Areas**

The program iterates through the `results` object, accessing the bounding boxes, class labels, and segmentation masks for each detected object. For objects classified as 'broiler', the segmentation mask is converted to a binary image, and the contours are extracted using OpenCV's `findContours` function.

Contour extraction is a crucial step in image processing and computer vision, as it allows for the identification and representation of object Contour extraction is a crucial step in image processing and computer vision, as it allows for the identification and representation of object boundaries. In this notebook, the contours are extracted from the binary segmentation mask, representing the precise outline of each detected broiler chicken.

The area of each contour is calculated using OpenCV's `cv2.contourArea` function, which computes the area enclosed by the contour. These individual contour areas are then summed to obtain the total segmented area for the corresponding broiler chicken. The segmented area serves as the primary input feature for the subsequent weight estimation step.

**4. Weight Estimation using Linear Regression**

The notebook/ program then loads a dataset containing the segmented areas and corresponding weights of broiler chickens (`area_weight.csv`). This dataset is crucial for training the linear regression model, as it provides the ground truth data necessary to establish the relationship between the segmented area (input feature) and the weight (target variable) of the chickens.

The linear regression model is trained using scikit-learn's `LinearRegression` class, a widely-used implementation of the linear regression algorithm in the Python ecosystem. The `train_test_split` function from scikit-learn is employed to split the dataset into training and testing subsets, enabling proper evaluation of the model's performance and generalization capabilities.

During the training process, the linear regression model learns the coefficients (slope and intercept) of the linear equation that best fit the training data. This equation can then be used to predict the weight of new broiler chickens based on their segmented areas obtained from the YOLOv8 model.

It is important to note that the accuracy and generalization performance of the weight estimation system heavily depend on the quality and representativeness of the training data used for the linear regression model. A diverse and well-curated dataset, capturing a wide range of broiler chicken sizes, breeds, and environmental conditions, is crucial for ensuring reliable and robust weight predictions.

**5. Evaluation and Results**

The notebook does not explicitly include an evaluation section, where the performance of the weight estimation system is quantitatively assessed. However, several evaluation metrics and techniques can be employed to measure the system's accuracy and identify potential areas for improvement.

a. **Mean Squared Error (MSE)**: The MSE is a commonly used metric for regression problems, measuring the average squared difference between the predicted weights and the actual weights in the test dataset. A lower MSE value indicates better prediction accuracy.

b. **Coefficient of Determination (R-squared)**: The R-squared value represents the proportion of the variance in the target variable (weight) that is explained by the input feature (segmented area). An R-squared value closer to 1 indicates a better fit of the linear regression model to the data.

c. **Residual Analysis**: Analyzing the residuals (the difference between the predicted and actual weights) can provide insights into the model's performance and potential violations of the linear regression assumptions. Residual plots can reveal patterns, heteroscedasticity (non-constant variance), or outliers that may require further investigation or model refinement.

d. **Cross-Validation**: Techniques such as k-fold cross-validation can be employed to assess the model's generalization performance and robustness to variations in the training data. This involves partitioning the dataset into multiple folds, training the model on a subset of folds, and evaluating its performance on the remaining folds.

e. **Visualization**: Visualizing the predicted weights against the actual weights can provide a intuitive understanding of the model's performance. Scatter plots or regression line plots can be used to identify potential outliers or non-linear patterns that may require more advanced regression techniques.