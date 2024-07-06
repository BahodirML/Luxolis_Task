import cv2
import numpy as np
import os
from mtcnn import MTCNN
from frames import frames
from frames import extract_frames

#detecting humans with their faces
def highlight_faces(frames):
    
    detector = MTCNN()

    # Create a directory 
    os.makedirs('humans', exist_ok=True)

    for i, frame in enumerate(frames):
        height, width, channels = frame.shape

        # Detect faces
        results = detector.detect_faces(frame)

        for result in results:
            if result['confidence'] > 0.5: 
                x, y, w, h = result['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imwrite(f'humans/highlighted_frame_{i+1}.jpg', frame)

# Extract frames
frames = extract_frames('human.mp4')

# Highlight faces 
highlight_faces(frames)

print("Processing complete. Check the highlighted directory for output images.")





