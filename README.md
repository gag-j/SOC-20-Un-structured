# SOC-20-Un-structured

This is a project I have undertaken as a part of Seasons of Code 2020, organised by the Web and Coding Club, IIT Bombay.

![](/cover.png)
## Project Description

The goal is to detect and recognize text from natural scene images - *text spotting*. The motivation behind the project can be found in this research paper: [Reading Text in the Wild](https://arxiv.org/pdf/1412.1842).

Broadly - the task can be divided into 2 parts - detection and recognition.

We employ the [YOLOv2](https://pjreddie.com/darknet/yolov2/) algorithm to perform the detection task. The original YOLOv2 research paper is available [here](https://arxiv.org/pdf/1612.08242.pdf). Candidate *bounding boxes* (rectangles that completely enclose text) are generated for each occurance of text in the image, which are cropped from the original image and fed to the recognition model.

A Convolutional Neural Network as described in the [Reading Text in the Wild](https://arxiv.org/pdf/1412.1842) is used to perfrom text recognition.

## Dataset

The [Google SVT](http://vision.ucsd.edu/~kai/svt/) dataset is used for overall training and evaluation. The dataset contains an image directory with the images, and XML files with the annotations for training and testing images.

The detection model is pre-trained on ImageNet and COCO datasets, we train the model on our Google SVT Training dataset.

The recognition model is trained using synthesized text images generated through this PyPI package [Text Recognition Data Generator](https://github.com/Belval/TextRecognitionDataGenerator)

## Text Detection
The text detection process can be summarized as follows:

1. [Initialization](yolo_initialize.ipynb) - Parse the annotation files using [The ElementTree XML API](https://docs.python.org/3/library/xml.etree.elementtree.html#module-xml.etree.ElementTree) and rearrange the dataset to create a desirable train:validation:test split. Save the annotations as a list of dictionaries with each dictionary containing relevant information about a particular image:
   - filename
   - height and width
   - lexicon 
   - coordinates of bounding box and text label for each occurance of text

2. [Generating anchor boxes](yolo_anchors.ipynb) - Generate anchor boxes using [k-means clustering](https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/). Anchor boxes are 'model rectangles' with different aspect ratios, each anchor box represents objects of a particular dimension. Using k-means, we group all our ground truth boxes into k anchor boxes

3. [Encoding input](yolo_preprocess_new.ipynb) - Preprocess the input image and encode the ground truth labels

4. [Helper Functions](yolo_utils_new.ipynb) - Write helper functions for decoding output of the model and displaying results

5. [Run the model](yolo_main.ipynb) - Design the model architecture, download the pretrained weights and start training. Decode the output and append the predicted bounding box coordinates to the list of dictionaries 

## Text Recognition

The text recognition process can be summarized as :

[Image-CNN](image_cnn.ipynb)

1. Concatenate the individual word lists of each image to create a master LEXICON for classification
2. Generate training images, editing background, skew and distortion as required using the Data Generator
3. Generate training and validation datasets
4. Design and train the Convolutional Neural Network as advised in [Reading Text in the Wild](https://arxiv.org/pdf/1412.1842), tune the model for best validation accuracy

[End-To-End](end_to_end.ipynb)

5. Use the above model to make predictions and append predicted labels to the list of dictionaries 

### Comments on the text recognition model:

- [x] Apply distortion, skew, and blur while generating text
- [ ] Add background, foreground and border coloring
- [ ] Incorporate [Google Fonts](https://github.com/google/fonts) into text generator for better training data
- [ ] Apply natural image blending while generating text     
