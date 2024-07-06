import cv2

def extract_frames(video_path, num_frames=30):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames

    frames = []
    for i in range(num_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = video.read()
        if ret:
            frames.append(frame)
            cv2.imwrite(f'frames/frame_{i+1}.jpg', frame)

    video.release()
    return frames

frames = extract_frames('human.mp4')




