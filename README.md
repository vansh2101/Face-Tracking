
# Face Tracking

Objective is to Design and implement a system for real-time face tracking in a video stream. The system should accurately detect and track faces with minimal latency.



## Table of Contents

* [Models Implemented](#models-implemented)
* [Enhancements](#enhancements)
* [Setup](#setup)


## Models Implemented
* **Mediapipe:** It is a cross-platform framework by Google that provides high-performance face detection and tracking

* **Haar Cascade Files:** These are data storage files for pre-trained Cascade Classifers.

* **YOLO:** An object detection model.

* **Single Shot Detector:** It is a deep learning model providing balance between speed and accuracy.

* **Multi-Task Cascaded Convolution Networks:** A deep learning method specifically designed for face detection.


## Enhancements
I optimised the YOLO model’s script to significantly reduce the time lag by introducing vectorized operations and Optical Flow technique to estimate the face movement rather than letting the model predict on each and every frame.


## Setup

To set up this project, 
* clone the repository:

```bash
git clone https://github.com/vansh2101/semantic-segmentation.git
cd semantic-segmentation
```

* Download the weights folder by clicking [here](https://drive.google.com/drive/folders/1PcuLiormZ3QGTbRrIgPuyAyDjEYZW-bt?usp=drive_link) and place it in the root directory.

* run the `optical_flow_detection.py` file.
    
