# Import necessary libraries
import os
import random
import shutil

# Set paths for your dataset
root_dir = r"C:\Users\sammy\PycharmProjects\YOLO\data_object_image_2"  # Path to the base directory of the KITTI dataset
images_dir = os.path.join(root_dir, 'training/image_2')  # Directory for the images
labels_dir = os.path.join(root_dir, 'KITTI_labels/label_2')  # Directory for the labels

# Directories for storing training and testing images/labels
train_images_dir = os.path.join(root_dir, 'train/images')
test_images_dir = os.path.join(root_dir, 'test/images')
train_labels_dir = os.path.join(root_dir, 'train/KITTI_labels')
test_labels_dir = os.path.join(root_dir, 'test/KITTI_labels')

# Create directories for training and testing if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# Get a list of all the image filenames (and label filenames)
image_files = sorted(os.listdir(images_dir))  # List of all image filenames
label_files = sorted(os.listdir(labels_dir))  # List of all label filenames

# Ensure the number of images and labels are the same
assert len(image_files) == len(label_files), "Number of images and labels do not match!"

# Define the split ratio for training and testing
split_ratio = 0.8  # 80% for training, 20% for testing
num_images = len(image_files)

# Shuffle the data for random splitting
indices = list(range(num_images))
random.shuffle(indices)

# Calculate split index
split_index = int(num_images * split_ratio)

# Split data into training and testing sets
train_indices = indices[:split_index]
test_indices = indices[split_index:]

# Move the training files
for idx in train_indices:
    # Move image files
    shutil.move(os.path.join(images_dir, image_files[idx]), os.path.join(train_images_dir, image_files[idx]))

    # Move corresponding label files
    shutil.move(os.path.join(labels_dir, label_files[idx]), os.path.join(train_labels_dir, label_files[idx]))

# Move the testing files
for idx in test_indices:
    # Move image files
    shutil.move(os.path.join(images_dir, image_files[idx]), os.path.join(test_images_dir, image_files[idx]))

    # Move corresponding label files
    shutil.move(os.path.join(labels_dir, label_files[idx]), os.path.join(test_labels_dir, label_files[idx]))

print("Data split complete: {} training images and {} testing images.".format(len(train_indices), len(test_indices)))
