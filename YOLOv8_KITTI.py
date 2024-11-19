# Import necessary libraries
from ultralytics import YOLO  # Import the YOLO class for object detection
import matplotlib.pyplot as plt  # Import Matplotlib for displaying images
import os  # Import os for file path manipulation

# Define paths to the KITTI dataset
KITTI_PATH = r"C:\Users\sammy\PycharmProjects\YOLO\data_object_image_2"  # Base directory of the KITTI dataset
TRAIN_IMAGES = os.path.join(KITTI_PATH, "train", "images")  # Directory containing training images
TEST_IMAGES = os.path.join(KITTI_PATH, "test", "images")  # Directory containing test images
DATA_CONFIG = os.path.join(KITTI_PATH, "data.yml")  # Path to the dataset configuration file
TRAIN_LABELS_PATH = os.path.join(KITTI_PATH, "train", "labels")  # Path for YOLO label files (training)
TEST_LABELS_PATH = os.path.join(KITTI_PATH, "test", "labels")  # Path for YOLO label files (testing)

# Define KITTI classes (object categories)
classes = ['Car', 'Pedestrian', 'Cyclist']  # Object classes in the KITTI dataset


# Function to test the pre-trained YOLO model on the KITTI dataset
def test_pretrained_yolo():
    print("Testing pre-trained YOLO model on KITTI...")

    # Load the pre-trained YOLO model (YOLOv8 nano variant is used here)
    model = YOLO("yolov8n.pt")  # You can replace with other variants like yolov8m.pt or yolov8l.pt

    # Run inference on the test images
    results = model.predict(source=TEST_IMAGES, save=True, conf=0.3)  # Confidence threshold set to 0.3

    print("Testing completed. Results saved in the runs/predict directory.")

    return results  # Return results for further analysis or visualization


# Function to evaluate the pre-trained YOLO model
def evaluate_pretrained_yolo():
    print("Evaluating pre-trained YOLO model...")

    # Load the pre-trained YOLO model
    model = YOLO("yolov8n.pt")  # Use YOLOv8 nano variant

    # Evaluate the model on the validation dataset using the data configuration
    results = model.val(data=DATA_CONFIG)

    # Output the evaluation metric: mAP at 0.5 IoU threshold
    print(f"Evaluation completed. mAP@0.5: {results.results_dict['metrics/mAP50(B)']}")


# Function to fine-tune the YOLO model on the KITTI training data
def fine_tune_yolo():
    print("Fine-tuning YOLO model on KITTI...")

    # Load the pre-trained YOLO model
    model = YOLO("yolov8n.pt")

    # Train the model on the KITTI training data
    model.train(data=DATA_CONFIG, epochs=10, imgsz=640, batch=16)

    print("Fine-tuning completed. Fine-tuned model saved in runs/train/weights.")


# Function to evaluate the fine-tuned YOLO model
def evaluate_fine_tuned_yolo():
    print("Evaluating fine-tuned YOLO model...")

    # Load the fine-tuned YOLO model using the best weights
    model = YOLO("runs/detect/train/weights/best.pt")

    # Evaluate the model on the validation dataset
    results = model.val(data=DATA_CONFIG)

    # Output the evaluation metric: mAP at 0.5 IoU threshold
    print(f"Evaluation completed. mAP@0.5: {results.results_dict['metrics/mAP50(B)']}")


# Function to test the fine-tuned YOLO model on the KITTI test dataset
def test_fine_tuned_yolo():
    print("Testing fine-tuned YOLO model on KITTI...")

    # Load the fine-tuned YOLO model
    model = YOLO("runs/detect/train/weights/best.pt")

    # Run inference on the KITTI test images
    results = model.predict(source=TEST_IMAGES, save=True, conf=0.3)  # Confidence threshold set to 0.3

    print("Testing completed. Results saved in the runs/predict directory.")

    return results  # Return results for further analysis or visualization


# Function to display inference results, including confidence scores
def display_results(results):
    print("Displaying results...")

    # Iterate over the first 10 results
    for result in results[:10]:
        # Get the annotated image (with bounding boxes drawn)
        annotated_img = result.plot()

        # Display the image using Matplotlib
        plt.imshow(annotated_img)
        plt.axis('off')  # Remove axes for a cleaner display
        plt.show()

    print("Results displayed.")


# Main script execution
if __name__ == "__main__":
    # Step 1: Test the pre-trained YOLO model on KITTI dataset
    results = test_pretrained_yolo()  # Run inference with the pre-trained YOLO model

    # Step 2: Display test results for visual inspection
    display_results(results)  # Call display_results to visualize predictions

    # Step 3: Evaluate the pre-trained YOLO model on the validation dataset
    evaluate_pretrained_yolo()  # Evaluate pre-trained model

    # Step 4: Fine-tune the YOLO model on the KITTI training data
    fine_tune_yolo()  # Fine-tune YOLO

    # Step 5: Evaluate the fine-tuned YOLO model
    evaluate_fine_tuned_yolo()  # Evaluate fine-tuned YOLO

    # Step 6: Test the fine-tuned YOLO model on the KITTI test dataset
    results = test_fine_tuned_yolo()  # Run inference with fine-tuned YOLO

    # Step 7: Display fine-tuned results for visual inspection
    display_results(results)  # Visualize fine-tuned predictions