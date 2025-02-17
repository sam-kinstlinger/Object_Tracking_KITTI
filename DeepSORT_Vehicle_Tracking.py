# Import necessary libraries
from ultralytics import YOLO  # YOLO is a popular object detection framework
import cv2  # OpenCV for image and video processing
import os  # For file and directory operations
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT for object tracking

# Define paths for the YOLO model, input image directory, and output directory
model_path = r"C:\Users\sammy\PycharmProjects\YOLO\yolov8n.pt"  # Path to the YOLOv8 model file
image_dir = r"C:\Users\sammy\PycharmProjects\YOLO\dataset\images"  # Path to the directory containing input images
output_dir = r"C:\Users\sammy\PycharmProjects\YOLO\dataset\output"  # Directory to save processed images

# Ensure the output directory exists; create it if it doesn't
os.makedirs(output_dir, exist_ok=True)

# Load the YOLOv8 model
yolo_model = YOLO(model_path)

# Initialize the DeepSORT tracker
# Parameters:
# - max_age: Maximum number of frames a track can go without being updated
# - nn_budget: Limits the size of the feature embedding queue for efficiency
deepsort = DeepSort(max_age=30, nn_budget=100)

# Dictionary to store the last known positions of tracks for velocity calculation
track_positions = {}

# Process each image in the input directory
for img_name in sorted(os.listdir(image_dir)):  # Sort images alphabetically for sequential processing
    img_path = os.path.join(image_dir, img_name)  # Get the full path to the image
    frame = cv2.imread(img_path)  # Read the image using OpenCV

    if frame is None:  # Handle cases where an image cannot be loaded
        print(f"Warning: Could not load image {img_path}. Skipping...")
        continue

    # Run YOLO model on the current frame to detect objects
    results = yolo_model.predict(source=frame, conf=0.4, verbose=False)

    # Convert YOLO detections to a format compatible with DeepSORT
    # YOLO outputs bounding boxes in the format [x1, y1, x2, y2], confidence, and class_id
    detections = []
    for det in results[0].boxes:  # Iterate through all detected boxes
        x1, y1, x2, y2, conf, cls = det.xyxy[0].tolist() + [det.conf[0].item()] + [det.cls[0].item()]
        # Convert bounding box to the required format for DeepSORT ([left, top, width, height])
        detections.append([[x1, y1, x2 - x1, y2 - y1], conf, int(cls)])

    # Update the DeepSORT tracker with the current detections
    # Tracks provide information about object identities across frames
    tracks = deepsort.update_tracks(detections, frame=frame)

    # Draw tracked bounding boxes, object IDs, and print state information
    for track in tracks:
        # Skip tracks that are not confirmed or have not been updated recently
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        # Retrieve the track ID and bounding box (in [x, y, width, height] format)
        track_id = track.track_id
        bbox = track.to_ltwh()  # Convert to [x, y, width, height]
        x_center = bbox[0] + bbox[2] / 2  # Calculate the x-coordinate of the center
        y_center = bbox[1] + bbox[3] / 2  # Calculate the y-coordinate of the center
        position = (x_center, y_center)  # Store the center position as (x, y)

        # Calculate velocity based on the difference in position between frames
        if track_id in track_positions:
            prev_position = track_positions[track_id]  # Retrieve the previous position
            velocity = (position[0] - prev_position[0], position[1] - prev_position[1])  # Calculate velocity
        else:
            velocity = (0, 0)  # Initial velocity for new tracks

        # Update the track's position in the dictionary
        track_positions[track_id] = position

        # Print the state: position and velocity
        print(f"Image: {img_name} | Track ID: {track_id} | Position: {position} | Velocity: {velocity}")

        # Draw bounding box and track ID on the frame
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save the annotated frame to the output directory
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, frame)  # Save the processed image

# Create a video from the processed frames in the output directory
output_video_path = r"C:\Users\sammy\PycharmProjects\YOLO\tracking_output.mp4"  # Path to save the video
fps = 2  # Frames per second for the output video
sample_frame = cv2.imread(os.path.join(output_dir, sorted(os.listdir(output_dir))[0]))  # Load a sample frame
frame_size = (sample_frame.shape[1], sample_frame.shape[0])  # Get the frame dimensions (width, height)

# Initialize the VideoWriter for saving the video
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
for img_name in sorted(os.listdir(output_dir)):  # Iterate through processed frames in order
    img_path = os.path.join(output_dir, img_name)  # Get full path to the frame
    frame = cv2.imread(img_path)  # Read the frame
    out.write(frame)  # Write the frame to the video
out.release()  # Finalize and save the video

# Visualize the tracking results in a window
for img_name in sorted(os.listdir(output_dir)):  # Iterate through processed frames
    img_path = os.path.join(output_dir, img_name)  # Get full path to the frame
    frame = cv2.imread(img_path)  # Read the frame
    cv2.imshow("Tracking", frame)  # Display the frame in a window
    if cv2.waitKey(30) & 0xFF == ord("q"):  # Break the loop if 'q' is pressed
        break
cv2.destroyAllWindows()  # Close all OpenCV windows