import cv2
import numpy as np
import os
from mtcnn import MTCNN
from frames import frames
from frames import extract_frames


def highlight_faces(frames):
    # Initialize MTCNN face detector
    detector = MTCNN()

    # Create directory if it doesn't exist
    os.makedirs('humans', exist_ok=True)

    for i, frame in enumerate(frames):
        height, width, channels = frame.shape

        # Detecting faces
        results = detector.detect_faces(frame)

        for result in results:
            if result['confidence'] > 0.5:  # You can adjust the confidence threshold
                x, y, w, h = result['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imwrite(f'humans/highlighted_frame_{i+1}.jpg', frame)

# Extract frames
frames = extract_frames('human.mp4')

# Highlight faces in the frames
highlight_faces(frames)

print("Processing complete. Check the highlighted directory for output images.")





