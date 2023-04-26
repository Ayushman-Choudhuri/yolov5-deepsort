import cv2
import time
import os 
import sys
import yaml

with open('config.yml' , 'r') as f:
    config =yaml.safe_load(f)['yolov5_deepsort']['main']

# Add the src directory to the module search path
sys.path.append(os.path.abspath('src'))

from src.detector import YOLOv5Detector
from src.tracker import DeepSortTracker
from src.dataloader import cap

YOLO_MODEL_NAME = 'yolov5n'

# Image Display Parameters
DISP_FPS = config['disp_fps'] 
DISP_OBJ_COUNT = config['disp_obj_count']
DISP_TRACKS = config['disp_tracks'] 
DISP_OBJ_TRACK_BOX = config['disp_obj_track_box']

object_detector = YOLOv5Detector(model_name=YOLO_MODEL_NAME)
tracker = DeepSortTracker()

track_history = {}    # Define a dictionary to store the previous center locations for each track ID

while cap.isOpened():

    success, img = cap.read() # Read the image frame from data source 
 
    start_time = time.perf_counter()    #Start Timer - needed to calculate FPS
    
    # Object Detection
    results = object_detector.score_frame(img)  # run the yolo v5 object detector 
    detections , num_objects= object_detector.extract_detections(results, img, height=img.shape[0], width=img.shape[1]) # Plot the bounding boxes and extract detections (needed for DeepSORT) and number of relavent objects detected


    # Object Tracking
    tracks = tracker.object_tracker.update_tracks(detections, frame=img)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id

         # Retrieve the current track location and bounding box
        location = track.to_tlbr()
        bbox = location[:4].astype(int)
        bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
         # Retrieve the previous center location, if available
        prev_center = track_history.get(track_id)

        # Draw the track line, if there is a previous center location
        if prev_center is not None:
            cv2.line(img, prev_center, bbox_center, (255, 0, 0), 2)
            

        cv2.rectangle(img,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),2)
        cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

         # Store the updated history for this track ID
        track_history[track_id] = bbox_center
        
    
    end_time = time.perf_counter()
    
    # FPS Calculation
    total_time = end_time - start_time
    fps = 1 / total_time


    # Descriptions on the image 
    cv2.putText(img, f'FPS: {int(fps)}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(img, f'MODEL: {YOLO_MODEL_NAME}', (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(img, f'TRACKED CLASS: {object_detector.tracked_class}', (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(img, f'TRACKER: {tracker.algo_name}', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(img, f'DETECTED OBJECTS: {num_objects}', (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    
    cv2.imshow('img',img)


    if cv2.waitKey(1) & 0xFF == 27:
        break


# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()