# yolov5-deepsort

![Python Version](https://img.shields.io/badge/python-3.9-blue)
![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0.0%2Bcu117-EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white&logoWidth=40)



This project is an open-source implementation of a real-time object tracking system based on the YOLOv5 and DeepSORT algorithms. This project aims to provide a solution for object tracking in videos, with the ability to track multiple objects simultaneously in real-time. The YOLOv5 model is used to detect objects in each frame, and then the DeepSORT algorithm is used to track these objects across multiple frames and associate them with unique IDs. The project is implemented in Python using the PyTorch deep learning framework.

<p align="center">

<img align="center" src="https://github.com/Ayushman-Choudhuri/yolov5-deepsort/blob/main/images/DeepSORT.png">

</p>

## Dependencies
You should install the following packages in your environment to run this project: 

* ultralytics 
``` bash
pip install ultralytics
```
* deep-sort-realtime 1.3.2
``` bash
pip install deep-sort-realtime
```
* [pytorch](https://pytorch.org/) - if you want CUDA support with pytorch upgrade the installation based on the CUDA version your system uses.  

The list of dependencies for the project can also be found on the [environment.yml](environment.yml) file.

To recreate the environment used to develop this project, you can also create a conda environment using the environment.yml file provided:
``` bash
conda env create -f environment.yml
```

This will create a new environment named stereo-visual-odometry-env with all the necessary packages installed. You can then activate the environment using the following command:

``` bash
conda activate yolov5-deepsort
```

Once the environment is activated, you can run the project and use the packages listed above.


## Directory Structure 

```bash
yolov5-deepsort
├── main.py
├── src
    ├── dataloader.py
    ├── detector.py
    └── tracker.py
├── data
├── environment.yml
├── config.yml
├── README.md


``` 


## Running the Project
#### Step 1: Assuming the dependencies have been installed and the environment activated ,clone the repository

``` bash
git clone https://github.com/Ayushman-Choudhuri/yolov5-deepsort

```

### Step 2: Setting data source
* Option 1: Webcam  
  - If the input video frame is a webcam, in the **config.yml** file ,change the **data_source** parameter in the **dataloader** section to "webcam".
  - Open the **config.yml** file and change the **webcam_id** to the one on your respective computer. You can list all the video devices in the **/dev** directory sorted by time in reverse order. To list them please use the following command. 
```bash
  ls -ltrh /dev/video*

``` 
* Option 3: Video File 
  - Input your video file in .mp4 format in the data directory (as shown in the directory structure)
  - Open the **config.yml** file and update the **data_path** parameter in the **dataloader** section

### Step 3: Run the main.py file 
```bash
python3 main.py

``` 
## Results 

The project was run on a openly available [video](https://pixabay.com/videos/people-commerce-shop-busy-mall-6387/) on pixabay showing people in a mall

<p align="center">
<img align="center" src="https://github.com/Ayushman-Choudhuri/yolov5-deepsort/blob/main/results/mall.gif">
</p>

The Project was also run on a openly available [video](https://pixabay.com/videos/cars-motorway-speed-motion-traffic-1900/) on pixabay showing cars on the road.

<p align="center">
<img align="center" src="https://github.com/Ayushman-Choudhuri/yolov5-deepsort/blob/main/results/cars.gif">
</p>

## Evaluation
DeepSORT is a multi-object tracking algorithm, so to judge its performance we need special metrics and benchmark datasets. We will be using CLEARMOT metrics to judge the performance of our DeepSORT on the [MOT17](https://motchallenge.net/results/MOT17/) dataset.ClearMOT is a framework for evaluating the performance of a tracker over different parameters. 

This evaluation will be conducted soon. 


## Known Issues

1. There is a bug in the tracker.py file due to which currently the configuration parameters are not getting read from the config.yml file. This will be resolved soon. 
2. Due to the nature of the YOLO algorithm , when people stand too close to each other in the video, the total number of detections are lower than expected. This shall be resolved in a future project with Fast RCNN. 

## Future Work 

1. Implementation of Fast RCNN for object detection. This would be slower than YOLOv5 but would be able to accurately detect people when they stand too close to each other in crowded areas. 
2. C++ Implementation
3. ROS implementation

## References

* [YOLO Algorithm](https://arxiv.org/abs/1506.02640)
* [SORT Algorithm](https://arxiv.org/abs/1703.07402)
* [DeepSORT code repository](https://github.com/nwojke/deep_sort)
* [DeepSORT explained](https://medium.com/augmented-startups/deepsort-deep-learning-applied-to-object-tracking-924f59f99104)
