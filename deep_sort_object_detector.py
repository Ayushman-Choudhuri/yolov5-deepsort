import cv2
import numpy as np
import sys
import glob

import time
import torch 

# YOLO V8 Parameters

# Deep Sort Parameters


# Data Source Parameters

DATA_SOURCE = "webcam"
WEBCAM_ID = 2 # if using external webcam. Please modify based on your system

DATA_PATH = "./data/people.mp4"
FRAME_RATE = 30

if DATA_SOURCE == "webcam": 

    cap = cv2.VideoCapture(WEBCAM_ID)  # external webcam ID = 2 

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        success, img = cap.read()
        if success:
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

elif DATA_SOURCE == "video file": 

    # Open the video file
    cap = cv2.VideoCapture(DATA_PATH)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error opening video file")

    # Read and display each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            cv2.imshow('Video', frame)

            # Exit if the 'q' key is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

else: 
    print("Input correct DATA SOURCE")

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
