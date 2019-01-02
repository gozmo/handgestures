import numpy as np
import cv2

def start_capture(frames):
    print("capturing frames")
    frames_count = 0
    captured_frames = []

    cap = cv2.VideoCapture(0)
    while(frames_count < frames):
        ret, rgb_frame = cap.read()
        bw_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

        captured_frames.append(bw_frame)
        frames_count += 1
        if frames_count % 10 == 0:
            print(frames_count)

    cap.release()
    return captured_frames
