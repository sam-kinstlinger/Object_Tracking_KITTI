# Import necessary libraries
import os

# Define paths to the KITTI dataset
KITTI_PATH = r"C:\Users\sammy\PycharmProjects\YOLO\data_object_image_2"
DATA_CONFIG = os.path.join(KITTI_PATH, "data.yml")
KITTI_TRAIN_LABELS_PATH = os.path.join(KITTI_PATH, "train", "KITTI_labels")
YOLO_TRAIN_LABELS_PATH = os.path.join(KITTI_PATH, "train", "labels")
KITTI_TEST_LABELS_PATH = os.path.join(KITTI_PATH, "test", "KITTI_labels")
YOLO_TEST_LABELS_PATH = os.path.join(KITTI_PATH, "test", "labels")

# Define KITTI classes (object categories)
classes = ['Car', 'Pedestrian', 'Cyclist']

# Function to create directories if they don't exist
def create_dirs(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

# Function to convert KITTI label file format to YOLO format
def convert_labels(label_files, label_path, yolo_path):
    for label_file in label_files:
        label_path_full = os.path.join(label_path, label_file)
        yolo_label_file = os.path.join(yolo_path, label_file)

        # Open KITTI label file and corresponding YOLO label file
        with open(label_path_full, 'r') as f_in, open(yolo_label_file, 'w') as f_out:
            for line in f_in:
                parts = line.split()
                class_name = parts[0]
                if class_name not in classes:
                    continue  # Skip non-relevant classes
                class_id = classes.index(class_name)
                x_min, y_min, x_max, y_max = map(float, parts[4:8])

                # Normalize the coordinates
                image_width, image_height = 1242, 375
                x_center = (x_min + x_max) / 2 / image_width
                y_center = (y_min + y_max) / 2 / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height

                # Write YOLO formatted label
                f_out.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Function to handle the label conversion for both train and test sets
def convert_kitti_to_yolo():
    # Create YOLO labels directories if they don't exist
    create_dirs([YOLO_TRAIN_LABELS_PATH, YOLO_TEST_LABELS_PATH])

    # Get label files for train and test sets
    label_files_train = os.listdir(KITTI_TRAIN_LABELS_PATH)
    label_files_test = os.listdir(KITTI_TEST_LABELS_PATH)

    # Convert labels for train and test sets
    convert_labels(label_files_train, KITTI_TRAIN_LABELS_PATH, YOLO_TRAIN_LABELS_PATH)
    convert_labels(label_files_test, KITTI_TEST_LABELS_PATH, YOLO_TEST_LABELS_PATH)

# Main function to run the entire process
def main():
    convert_kitti_to_yolo()  # Convert KITTI labels to YOLO format

if __name__ == "__main__":
    main()