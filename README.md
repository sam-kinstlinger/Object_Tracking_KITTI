# KITTI_YOLO
This project demonstrates how to retrain a YOLOv8 model using the KITTI dataset for object detection tasks. The KITTI dataset is a widely-used benchmark for autonomous driving research, including rich annotations of objects such as cars, pedestrians, and cyclists. By fine-tuning YOLOv8 on this dataset, we aim to improve its performance on tasks specific to autonomous vehicle perception systems. This project includes steps to prepare the dataset, convert annotations to YOLO format, split the data into training and testing sets, fine-tune the YOLO model, and evaluate its performance.

**Files**
- Split_Dataset.py: Splits KITTI dataset into training and testing subsets
- Convert_Labels.py: Converts KITTI annotations into YOLO-compatible format.
- YOLOv8_KITTI.py: Tests pre-trained YOLOv8 on KITTI dataset, Fine-tunes YOLOv8 model using KITTI training data, evaluates model performance and displays results
- Re-trained_YOLO_Image.jpg contains a visualized example of inference performed by the retrained model on an image in the KITTI Dataset
- data.yml: The dataset configuration definition file to specify paths, the number of classes, and class names

**Steps**
- Install necessary libraries
- Download KITTI Dataset
- Run Split_Dataset.py
- Run Convert_Labels.py
- Run YOLOv8_KITTI.py
- Results will be saved in the runs/predict directory for visualization
- The fine-tuned model and weights will be saved in the runs/train/weights directory
