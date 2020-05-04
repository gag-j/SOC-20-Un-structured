# SOC-20-Un-structured

This is a project I am mentoring as a part of Seasons of Code 2020, organised by Web and Coding Club, IIT Bombay

![](/cover_1.png)

## Project Description

We all have used scanned copies of books and have been irritated by the fact that we cannot directly navigate to a particular section of the document or do a Ctrl+F. This project aims to take unstructured text as the input data and to give us nice and good looking structured text. This is not only restricted to properly written text as in a book but also extends to targeting problems like a self driving car detecting and understanding random road signs, automatic detection systems to record and interpret number plates of vehicles which did not follow the red light, and much more!

So, let's dwell deep into the technicalities of the project. In simple words, the problem we're targeting is as follows - "You have an image as an input which contains text at certain locations in the image. You need to detect the text and generate the text present in the input as the output. This involves segmenting the given image to focus only on the part of the image which contains text. Then you need to implement text recognition to generate the output."

## Initial References 

* [CPP](https://www.learncpp.com/) (although we won’t be using CPP it’s a good idea to refresh)
* [Python](https://www.learnpython.org/)
* [Python Practice](https://www.hackerrank.com/domains/python)
* [Deep Learning](https://www.coursera.org/specializations/deep-learning)
* [Intro to OCR](https://towardsdatascience.com/a-gentle-introduction-to-ocr-ee1469a201aa)

## Rough Timeline

* Week 1 - learn the basics of version control (GitHub). Go through the shared article properly to get a better idea of what is required. Try out some basic problems on Hackerrank to brush up your Python programming skills.
* Week 2 - Install Ubuntu and set up a development environment. Meanwhile, you can continue working on your Python skills.
* Week 3 - Learn about Linux commands, working with NumPy, Jupyter. Start learning about Neural Networks.
* Week 4 - Continue learning about Deep Learning architectures. Start looking for good datasets to train and test the model.
* Week 5 - Learn concepts of Image Classification and Recognition. Try simple implementations for the same.
* Week 6+7 - Start working on the chosen dataset. Build a deep learning network that classifies and recognizes the data properly.
* Week 8 - Work on tuning and training of the network on different datasets and analyze the results.
* Week 9 - Buffer Week. Keep experimenting with the model to optimize results and improve accuracy.
* Week 10 - Properly document the project. Write a short description of your project experience. 

## Checkpoints

* Checkpoint 1 - Linux environment setup and basic proficiency in python.
* Checkpoint 2 - Basics of shell commands, and machine learning environment. Concepts of neural networks.
* Checkpoint 3 - Clarity about image classification and recognition tasks. Choose datasets for the project.
* Checkpoint 4 - Basic model ready which does the job with some amount of accuracy.
* Checkpoint 5 - Parameter tuning leading to improvement in the performance of the model.

## Teams

Team 1 | Team 2 | Team 3 | Team 4
------------ | ------------- | ------------- | ------------- 
Ritwik | Suraj | Aryan | Aman
Deepanshu | Supreet | Param | Manan
Heetak | Vaishnavi | Nabarun | Anupam
. | . | Sudheeradh | .
## Weekly Targets

### Week 1 (22nd March - 28th March)

Start off by refreshing your python programming skills. Try solving some questions on websites like Hackerrank, etc. This way, you'll get familiar with the basic syntax. And for the rest, you can always google as and when required. Also, learn how to work with GitHub, create a profile if you don’t have one, and learn basic operations like pull, push, merge, etc. You can create a repository and push your python practice codes to that.
Also, I'd recommend setting up Ubuntu as it's much more convenient but it is up to you. For converting the problem to code, you'll work on Google Collab so it doesn't matter much.
I suggest working in teams of 3 each. You can choose the teams for yourself. 

Please share your Github account details so that I can add you to the Project Repository.

I have created teamwise branches in the repository where you will push your codes and commit your changes to the codes. Having the entire project in one repository will make it easy to manage. 

A guide that helps in the structuring of the project: [Project Structuring](https://docs.python-guide.org/writing/structure/)

I expect a well structured (hence the project name :p) and nicely documented project. Go through this as well. A lot of companies expect this kind of documentation for version-controlled projects that you’ll do in your life ahead.

So, basic targets for this Week  -

* Brush up your Python programming skills.
* Get familiar with GitHub
* Go through the documentation strategies.
* (Headstart for Week 2) Start brainstorming about how to go about various components of the project.

### Week 2 (29th March - 4th April)

So, having gone through the basics of python and some documentation stuff, now you should start exploring the area of Deep Learning. FYI Deep Learning (referred to as DL from now on) is a subset of Machine Learning (ML), but for our purpose, we’ll directly jump to the DL part. You are encouraged to look at the traditional Machine Learning techniques as well if you have time. So, I’d recommend starting with the DL specialization by Andrew NG on Coursera (I’ve shared the link earlier). You are expected to go through the first four courses. The first course introduces you to the cool things you’ll be doing. The second and third courses will teach you a lot of concepts (read life hacks) to improve your performance. The fourth course is the most important one, which will help you understand Convolutional Neural Networks (CNNs) which we will be frequently using. While going through the course assignments, pay attention to small details (coz a lot of code is already written there and you just have to fill in stuff). 

Other than this, I have shared a paper that uses text recognition on GitHub. Go through it properly. Give it a quick read before you start learning DL and mark the points/terms you do not understand. After you finish off your learning, you will start working on implementing the paper (probably two weeks from now)

So, basic targets for the Week - 

* Go through the paper which implements text recognition nicely
* Start learning DL (at least cover the first two courses in this week. Don’t think much about the week-wise distribution of * content given in the courses. It’s a scam!)

### Week 3 (5th April - 11th April)

Continue with the DL specialization and complete the planned content. Also, start working on the standard classification problem of handwritten digits (MNIST dataset) once you are done with the courses. This will give you a feel of what it is like to implement a network architecture from scratch and what things you should keep in mind. The model should be capable of achieving an accuracy of at least 95% (perfectly doable, don’t worry). Also, go through the reference literature again. 

Here is a link to the digit classification problem: [Handwritten Digit Classification](https://www.kaggle.com/c/digit-recognizer/overview)

So, basic targets for the Week -

* Complete the DL specialization (Course 3 and Course 4)
* Write an image classification network architecture for handwritten for the MNIST dataset.

### Week 4 (12th April - 18th April)

Some of you still have your courses going on, so for this week, the targets are as follows -
 * Complete the courses 
 * Train a model for the MNIST task mentioned earlier. Try to change the hyperparams and network architechure to maximize accuracy. Work in your teams of 3 to do this. Push the code in your individual team branches by this Saturday.
 * Re-read the reference literature on the problem
 * After reading the paper, start working on the detection task as mentioned in the paper. Use the street view dataset for the task as used in the paper. 
 
 These tasks are mentioned in the decreasing order of priority for this week. By the end of the week, make sure that you have completed atleast the first three tasks which are mentioned here.
 
 Some resources to get started with the project -
Data Pre-processing:
Get started with this: 
* [1](https://towardsdatascience.com/image-pre-processing-c1aec0be3edf)
* [2](https://medium.com/cashify-engineering/improve-accuracy-of-ocr-using-image-preprocessing-8df29ec3a033)
* [3](https://www.researchgate.net/publication/221909023_Preprocessing_Techniques_in_Character_Recognition)

This should suffice, i guess. Once you start building your model, you will realize how to go about things and implement them. Also, one early suggestion, when you reach the training stage, make sure that your model works, i.e., training starts without any errors on a smaller portion of a dataset (say 10 images). See that you get some values of loss and accuracy, no matter whatever they are. Once you are sure of this, import the complete dataset and start training. You can also use the same trick while trying to improve your model by tuning.

Here are some resources for the Bounding Box algorithm -
* [bounding box](https://pdollar.github.io/files/papers/ZitnickDollarECCV14edgeBoxes.pdf)
* [implementation](https://github.com/pdollar/edges)
* [blog](https://blog.csdn.net/wsj998689aa/article/details/39476551) You'll have to use google translate :p

* [YOLO](https://medium.com/analytics-vidhya/yolo-v3-theory-explained-33100f6d193)

* Learn the concept of IOU (intersection over union). We'll use that as an evaluative metric.


### Week 5 (19th April - 25th April)

Okay so I am assuming that all of you have completed the courses. Some of you have still not completed your MNIST problem. I urge you to do that as soon as possible. Not only because the problem is a simpler version of our project target, but also because training a ML model from scratch gives you an exposure to the various different programming practices. 

Talking about the week's targets, you are now required to work in teams and on every branch, there should be a single file for a specific task. You are encouraged to divide tasks among yourselves. The major target for this week is to complete the data loading part for the project. I assume that you have gone through the paper which we will be implementing.  By the end of the week, you should be completely ready to build a training model and start training. This will require proper data preprocessing, ensuring proper input to the network. The dataset which we will use for the project is the same which was mentioned in the paper (Street View). 

I have already shared some resources for your help. In case of doubts, give a try to solve the problems by yourself. Google things and understand potential solutions to your problems. And if you don't find a solution, feel free to post on the group.

### Week 6 (26th April - 2nd May)

Ok, after you finish your data loading and pre-processing part, you can start implementing the model for training. In case you're still lagging behind in your data part, you should be complelety done with that part by Tuesday. As I also mentioned on the group, you can also try different architechures for your training and see how they work, but the basic target for this week is that every team should be ready with atleast one working training model. For now, it is fin if the accuracy is not good enough, we will dedicate time in the next week for fine-tuning the model as well. In case you finish early with the model, you can start hyperparams tuning for your model. Some resources are already posted (under WEEK 4). For specific doubts, try to google them and try to resolve them by yourselves. If in case you're still stuck, tell me ASAP. Note that there is no issue if you directly ask me doubts, but "Googling" is an important skill in programming, so also try figuring out things by yourselves :)


### Week 7 (3th May - 9th May)

So, a lot of you haven't completed the training part completely, so continue working on that. Some of you are finding it easier to work with YOLO, so you can proceed with that. Keep it a target to complete the training with a good accuracy by the end of the week. Needless to say, but work on a train-test split so that you can validate your model quantitatively. In case you complete this early, the next task will be to start working on Recognition. Try exploring methods for recognition. Note that this is only after you complete the detection part, and is formally the task for next week.


