# yolov5-deepsort

This project is an open-source implementation of a real-time object tracking system based on the YOLOv5 and DeepSORT algorithms. This project aims to provide a solution for object tracking in videos, with the ability to track multiple objects simultaneously in real-time. The YOLOv5 model is used to detect objects in each frame, and then the DeepSORT algorithm is used to track these objects across multiple frames and associate them with unique IDs. The project is implemented in Python using the PyTorch deep learning framework.

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

## Running the Project

## Results 

## Known Issues

## Future Work 

## References

* [YOLO Algorithm](https://arxiv.org/abs/1506.02640)
* [SORT Algorithm](https://arxiv.org/abs/1703.07402)
* [DeepSORT code repository](https://github.com/nwojke/deep_sort)
