# KITTI_YOLO
This project demonstrates how to retrain a YOLOv8 model using the KITTI dataset for object detection tasks. It includes steps to prepare the dataset, convert annotations to YOLO format, split the data into training and testing sets, fine-tune the YOLO model, and evaluate its performance.

**Files**
- Split_Dataset.py: Splits KITTI dataset into training and testing subsets
- Convert_Labels.py: Converts KITTI annotations into YOLO-compatible format.
- YOLOv8_KITTI.py: Tests pre-trained YOLOv8 on KITTI dataset, Fine-tunes YOLOv8 model using KITTI training data, evaluates model performance and displays results

**Steps**
- Install necessary libraries
- Download KITTI Dataset
- Run Split_Dataset.py
- Run Convert_Labels.py
- Run YOLOv8_KITTI.py
