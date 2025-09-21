import cv2 # type: ignore
import os
import numpy as np # type: ignore
from concurrent.futures import ThreadPoolExecutor

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extracts frames from a video at a specified frame rate.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory where extracted frames will be saved.
        frame_rate (int): Extract one frame every 'frame_rate' seconds.

    Returns:
        List of file paths of extracted frames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame at the specified frame rate
        if frame_count % int(cap.get(cv2.CAP_PROP_FPS) * frame_rate) == 0:
            frame_filename = f"{output_dir}/frame_{frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)
            saved_frames.append(frame_filename)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return saved_frames

def preprocess_frame(frame_path, output_size=(224, 224)):
    """
    Preprocesses a frame by resizing and normalizing it.

    Args:
        frame_path (str): Path to the image frame.
        output_size (tuple): Desired output size for the frame.

    Returns:
        Numpy array of the preprocessed frame.
    """
    frame = cv2.imread(frame_path)
    frame = cv2.resize(frame, output_size)
    frame = frame / 255.0  # Normalize to range [0, 1]
    return frame

def batch_preprocess_frames(frame_paths, output_dir, batch_size=32):
    """
    Batch processes frames for model input.

    Args:
        frame_paths (list): List of file paths to the frames.
        output_dir (str): Directory where preprocessed frames will be saved.
        batch_size (int): Number of frames to process in one batch.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(0, len(frame_paths), batch_size):
        batch = frame_paths[i:i+batch_size]
        preprocessed_batch = []

        with ThreadPoolExecutor() as executor:
            for frame in executor.map(preprocess_frame, batch):
                preprocessed_batch.append(frame)

        batch_filename = os.path.join(output_dir, f"batch_{i//batch_size}.npy")
        np.save(batch_filename, np.array(preprocessed_batch))

# Example usage
video_path = 'path_to_video.mp4'
output_frames_dir = 'extracted_frames'
output_batches_dir = 'preprocessed_batches'

# Extract frames from video
frames = extract_frames(video_path, output_frames_dir, frame_rate=1)

# Preprocess frames in batches
batch_preprocess_frames(frames, output_batches_dir, batch_size=32)
