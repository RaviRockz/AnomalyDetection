# Anomaly Detection
Detecting the circulation of non-pedestrian entities include bikers, skaters, cart in the campus walkways using UCSD anomaly detection dataset
with Faster R-CNN based object detection technique.
## Dependencies
* python 3.6.9
* ubuntu 18.0.3 LTS
* python packages
    * tensorflow
    * opencv-python
    * [faster r-cnn inception v2 coco object detection model from model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
## How To Run
* python3 detect_on_image.py to detect anomaly on image
* python3 detect_on_video.py to detect anomaly on video
## Input and Output
    <p>
    <img width=360 height=240 src="doc_data/086_input.tif" alt="">
    </p>
    <p>
    <img width=360 height=240 src="doc_data/086_output.tif" alt="">
    </p>