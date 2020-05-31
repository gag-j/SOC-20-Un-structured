# SOC-20-Un-structured

This is a project I am undertaking as a part of Seasons of Code 2020, organised by Web and Coding Club, IIT Bombay
![](/cover.png)

## Project Description

The goal is to detect and recognize text from natural images - text spotting. The motivation behind the project can be found in this [research paper](https://arxiv.org/pdf/1412.1842).

Broadly - the task can be divided into 2 parts - detection and recognition.
We employ the [YOLOv2](https://pjreddie.com/darknet/yolov2/) algorithm to perform the detection task. The original YOLOv2 research paper is available [here](https://arxiv.org/pdf/1612.08242.pdf). We generate candidate *bounding boxes* (rectangles that completely enclose) for each occurance of text in the image, which we crop from the original image and feed to the recognition model.
We then use a Convolutional Neural Network as described in the above [research paper](https://arxiv.org/pdf/1412.1842) to perfrom text recognition.

## Dataset

The [Google SVT](http://vision.ucsd.edu/~kai/svt/) dataset is used for overall training and evaluation. The dataset contains an image directory with the images, and training and testing xml files with the annotations.

The detection model is pre-trained on ImageNet and COCO datasets, we train the model on our Google SVT Training dataset.
The recognition model is trained using synthesized text images generated through this pip installable [Text Recognition Data Generator](https://github.com/Belval/TextRecognitionDataGenerator)

### Detection

The detection process can be summarized as follows:
1. [Initialization](yolo_initialize.ipynb) - Parse the annotation files using [The ElementTree XML API](https://docs.python.org/3/library/xml.etree.elementtree.html#module-xml.etree.ElementTree) and rearrange the dataset to create a desirable train:validation:test split. Save the annotations as a list of dictionaries with each dictionary containing relevant information:
-filename
-height and width
-lexicon 
-coordinates of bounding box and text label for each occurance of text
about a particular image.

2. [Generating anchor boxes](yolo_anchors.ipynb) - Generate anchor boxes using [k-means clustering](https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/). Anchor boxes are 'model rectangles' with different aspect ratios, each anchor box represents objects of a particular dimension. Using k-means, we group all our ground truth boxes into k anchor boxes.

3. [Encoding input](yolo_preprocess_new.ipynb) - Preprocess the input image and encode the ground truth labels.

4. [Helper Functions](yolo_utils_new.ipynb) - Write helper functions for decoding output of the model and displaying results.

5. [Run the model](yolo_main.ipynb) - Design the model architecture, download the pretrained weights and start training. Decode the output and append the predicted bounding box coordinates to the list of dictionaraies 

## Recognition

The recognition process can be summarized as :
