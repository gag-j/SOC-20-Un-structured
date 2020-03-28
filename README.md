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
* Re-read the reference literature on the problem
* Write an image classification network architecture for handwritten for the MNIST dataset.






