import cv2
import numpy as np
import torch 

class YOLOv5Detector(): 

    def __init__(self, model_name):

        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: " , self.device)

        self.downscale_factor = 1  # Reduce the resolution of the input frame by this factor to speed up object detection process
        self.confidence_threshold = 0.1 # Minimum theshold for the detection bounding box to be displayed
        self.tracked_class = 'person'

    def load_model(self , model_name):  # Load a specific yolo v5 model or the default model

        if model_name: 
            model = torch.hub.load('ultralytics/yolov5' , 'custom' , path = model_name , force_reload = True)
        else: 
            model = torch.hub.load('ultralytics/yolov5' , 'yolov5s' , pretrained = True)
        return model
 
    def score_frame(self , frame): 
        self.model.to(self.device) # Transfer a model and its associated tensors to CPU or GPU
        frame_width = int(frame.shape[1]/self.downscale_factor)
        frame_height = int(frame.shape[0]/self.downscale_factor)
        frame_resized = cv2.resize(frame , (frame_width,frame_height))

        yolo_result = self.model(frame_resized)

        labels , bb_cord = yolo_result.xyxyn[0][:,-1] , yolo_result.xyxyn[0][:,:-1]
        
        return labels , bb_cord
        

    def class_to_label(self, x):

        return self.classes[int(x)]
        
    def plot_boxes(self, results, frame, height, width):

        labels, bb_cordinates = results  # Extract labels and bounding box coordinates
        detections = []         # Empty list to store the detections later 
        class_count = 0
        num_objects = len(labels)
        x_shape, y_shape = width, height

        for object_index in range(num_objects):
            row = bb_cordinates[object_index]

            if row[4] >= self.confidence_threshold:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                
                if self.class_to_label(labels[object_index]) == self.tracked_class :
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    x_center = x1 + ((x2-x1)/2)
                    y_center = y1 + ((y2 - y1) / 2)
                    conf_val = float(row[4].item())
                    feature = self.tracked_class

                    class_count+=1
                    
                    detections.append(([x1, y1, int(x2-x1), int(y2-y1)], row[4].item(), feature))
                    # We structure the detections in this way because we want the bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class) - Check deep-sort-realtime 1.3.2 documentation

        return frame , detections , class_count