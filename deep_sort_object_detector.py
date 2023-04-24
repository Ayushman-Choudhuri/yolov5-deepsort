import cv2
import numpy as np
import sys
import glob
import os
import time
import torch 
from deep_sort_realtime.deepsort_tracker import DeepSort


# YOLO Parameters
DOWNSCALE_FACTOR = 1  # Reduce the resolution of the input frame by this factor to speed up object detection process
CONFIDENCE_THRESHOLD = 0.1 # Minimum theshold for the bounding box to be displayed
YOLO_MODEL_NAME = 'yolov5n'
TRACKED_CLASS = 'person'

# Deep Sort Parameters

MAX_AGE = 5                 # Maximum number of frames to keep a track alive without new detections. Default is 30

N_INIT =1                  # Minimum number of detections needed to start a new track. Default is 3

NMS_MAX_OVERLAP = 1.0       # Maximum overlap between bounding boxes allowed for non maximal supression(NMS).
                            #If two bounding boxes overlap by more than this value, the one with the lower confidence score is suppressed. Defaults to 1.0.

MAX_COSINE_DISTANCE = 0.3   # Maximum cosine distance allowed for matching detections to existing tracks. 
                            #If the cosine distance between the detection's feature vector and the track's feature vector is higher than this value, 
                            # the detection is not matched to the track. Defaults to 0.2

NN_BUDGET = None            # Maximum number of features to store in the Nearest Neighbor index. If set to None, the index will have an unlimited budget. 
                            #This parameter affects the memory usage of the tracker. Defaults to None.

OVERRIDE_TRACK_CLASS = None  #Optional override for the Track class used by the tracker. This can be used to subclass the Track class and add custom functionality. Defaults to None.
EMBEDDER = "mobilenet"       #The name of the feature extraction model to use. The options are "mobilenet" or "efficientnet". Defaults to "mobilenet".
HALF = True                  # Whether to use half-precision floating point format for feature extraction. This can reduce memory usage but may result in lower accuracy. Defaults to True
BGR = False                   #Whether to use BGR color format for images. If set to False, RGB format will be used. Defaults to True.
EMBEDDER_GPU = True          #Whether to use GPU for feature extraction. If set to False, CPU will be used. Defaults to True.
EMBEDDER_MODEL_NAME = None   #Optional model name for the feature extraction model. If not provided, the default model for the selected embedder will be used.
EMBEDDER_WTS = None          # Optional path to the weights file for the feature extraction model. If not provided, the default weights for the selected embedder will be used.
POLYGON = False              # Whether to use polygon instead of bounding boxes for tracking. Defaults to False.
TODAY = None                 # Optional argument to set the current date. This is used to calculate the age of each track in days. If not provided, the current date is used.

# Data Source Parameters
DATA_SOURCE = "video file"   # source can be set to either "video file" or "webcam"
WEBCAM_ID = 2                # if using external webcam. Please modify based on your system
DATA_PATH = "./data/people.mp4" # path to the video file if you are using a video as input

# Image Display Parameters
DISP_FPS = True 
DISP_OBJ_COUNT = True
DISP_TRACKS = True 
DISP_OBJ_DETECT_BOX = True
DISP_OBJ_TRACK_BOX = True

############################################### Object Detector Class Definitions ###############################################

class YOLOv5Detector(): #yet to upgrade to v8

    def __init__(self, model_name):

        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: " , self.device)


    def load_model(self , model_name):  # Load a specific yolo v5 model or the default model

        if model_name: 
            model = torch.hub.load('ultralytics/yolov5' , 'custom' , path = model_name , force_reload = True)
        else: 
            model = torch.hub.load('ultralytics/yolov5' , 'yolov5s' , pretrained = True)
        return model
 
    def score_frame(self , frame): 
        self.model.to(self.device) # Transfer a model and its associated tensors to CPU or GPU
        frame_width = int(frame.shape[1]/DOWNSCALE_FACTOR)
        frame_height = int(frame.shape[0]/DOWNSCALE_FACTOR)
        frame_resized = cv2.resize(frame , (frame_width,frame_height))

        yolo_result = self.model(frame_resized)

        labels , bb_cord = yolo_result.xyxyn[0][:,-1] , yolo_result.xyxyn[0][:,:-1]
        
        return labels , bb_cord
        

    def class_to_label(self, x):

        return self.classes[int(x)]
        
    def plot_boxes(self, results, frame, height, width, confidence=CONFIDENCE_THRESHOLD):

        labels, bb_cordinates = results  # Extract labels and bounding box coordinates
        detections = []         # Empty list to store the detections later 
        class_count = 0
        num_objects = len(labels)
        x_shape, y_shape = width, height

        for object_index in range(num_objects):
            row = bb_cordinates[object_index]

            if row[4] >= confidence:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                
                if self.class_to_label(labels[object_index]) == TRACKED_CLASS :
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    x_center = x1 + ((x2-x1)/2)
                    y_center = y1 + ((y2 - y1) / 2)
                    conf_val = float(row[4].item())
                    feature = TRACKED_CLASS

                    class_count+=1
                    
                    detections.append(([x1, y1, int(x2-x1), int(y2-y1)], row[4].item(), feature))
                    # We structure the detections in this way because we want the bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class) - Check deep-sort-realtime 1.3.2 documentation

        return frame , detections , class_count
    

#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

####################################### Object Tracker Class Instantiation ################################################
object_tracker = DeepSort(max_age=MAX_AGE,
                n_init=N_INIT,
                nms_max_overlap=NMS_MAX_OVERLAP,
                max_cosine_distance=MAX_COSINE_DISTANCE,
                nn_budget=NN_BUDGET,
                override_track_class=OVERRIDE_TRACK_CLASS,
                embedder=EMBEDDER,
                half=HALF,
                bgr=BGR,
                embedder_gpu=EMBEDDER_GPU,
                embedder_model_name=EMBEDDER_MODEL_NAME,
                embedder_wts=EMBEDDER_WTS,
                polygon=POLYGON,
                today=TODAY)

# Select Data Source 
if DATA_SOURCE == "webcam": 
    cap = cv2.VideoCapture(WEBCAM_ID)
elif DATA_SOURCE == "video file": 
    cap = cv2.VideoCapture(DATA_PATH)
else: print("Enter correct data source")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


object_detector = YOLOv5Detector(model_name=YOLO_MODEL_NAME)

track_history = {}    # Define a dictionary to store the previous center locations for each track ID

while cap.isOpened():

    success, img = cap.read() # Read the image frame from data source 
 
    start_time = time.perf_counter()    #Start Timer - needed to calculate FPS
    
    results = object_detector.score_frame(img)  # run the yolo v5 object detector 

    # output = [[results[0], results[1]]]
    # print(output)

    img , detections , num_objects= object_detector.plot_boxes(results, img, height=img.shape[0], width=img.shape[1], confidence=0.5) # Plot the bounding boxes and extract detections (needed for DeepSORT) and number of relavent objects detected
    
    #print(detections ,"\n" )

    tracks = object_tracker.update_tracks(detections, frame=img)

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
    cv2.putText(img, f'TRACKED CLASS: {TRACKED_CLASS}', (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(img, f'DETECTED OBJECTS: {num_objects}', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    
    cv2.imshow('img',img)


    if cv2.waitKey(1) & 0xFF == 27:
        break


# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()