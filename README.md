# TrafficSignRecognition
Traffic Sign Recognition Using Deep Learning on Self-Driving Cars
## Introduction
Self-driving cars are autonomous decision-making systems. They process data from sensors such as cameras, LiDAR, RADAR, GPS, or inertia sensors [1], and the data is then loaded to internal systems where predictions are made. Then, the car can react accordingly to the street conditions identified. Self-driving cars are driving much more attention than before, as it is an energy-efficient and labor-saving approach to transportation [2].

This project investigates how image recognition is used in self-driving cars, which serves to be the first step towards self-driving. The resultant deep learning model will make predictions to the traffic sign images input and categorize them into correct classes. Thus, the predictions could be used for further mechatronic design of the driving system of the car. 

Deep CNN is a suitable tool to achieve our goal, as the algorithms allow the automatic detection of desired objects and has excellent functionality in categorizing common features in “similar” images, observing trends from mass input, and predicting the result into multiple classes. Thus, the system could “learn” better after each training process and as the number of samples increased. Figure 1 shows the general process plan of the project.
![image](https://user-images.githubusercontent.com/99308255/163247303-075647ff-932c-4967-adf2-e9fa824f94c8.png)

<img width="452" alt="image" src="https://user-images.githubusercontent.com/99308255/163247321-a6404fbd-96f5-4009-87d6-9e31f57b9ded.png">
Figure 1. Process illustration![image](https://user-images.githubusercontent.com/99308255/163247369-0979d98c-0673-4fff-a186-2a2c6550aa6f.png)

## Background & Related Works![image](https://user-images.githubusercontent.com/99308255/163247447-15dfcee9-1126-45d2-aa17-74d2d4d5d315.png)

Many researchers have determined traffic sign recognition as the initial step for this process. This technique is generally divided into two parts, one is the detection which involves the use of Yolo, the other is the classification in a research paper conducted 2019 [3]. We are borrowing the same thought map and focusing on the development of traffic sign classification in deep CNN. The image segmentation and detection are done using an open-source code using YOLOv5from GitHub to make our piece complete and more useful in a general form [4]. YoloV5 is introduced in a research paper used in segmentation step [5] and traffic signs are cropped out for our model testing purposes. Since the requirement for training the dataset needs to be large and accurate. The training and validation dataset is obtained from CTS (Chinese Traffic Signs) [6].![image](https://user-images.githubusercontent.com/99308255/163247499-dc0d128d-716b-482b-b3ab-4087214794d5.png)


### Data Processing ![image](https://user-images.githubusercontent.com/99308255/163247537-6da5f9b6-6ab5-43e7-aa79-d10d9ebec603.png)
The data processing of the project consists of two steps: the first stage was to adapt an external online dataset for training and the second was to generate and collect our own dataset. The online source will serve as the training set and our own generated data will be used as the validation and testing set.
During the first stage, the team obtained data from the Chinese Traffic Sign Database [6], which was supported by National Nature Science Foundation of China. The database consists of accurately cropped and labeled data where the classes are represented by numbers to increase efficiency. Figure 2 shows some of the sample data.

 <img width="252" alt="image" src="https://user-images.githubusercontent.com/99308255/163247616-41459ff9-1747-4062-99dc-a650826fc8e2.png">

Figure 2. Kaggle Sample Data
 
To generate data by us, the team collected the street images in front of the university using phone camera and manually crop them into desired size. Image segmentation technique was applied to do the automatic detection and cropping. Figure 3 shows an example of the raw data. 

 <img width="135" alt="image" src="https://user-images.githubusercontent.com/99308255/163247651-79cccacd-b3f4-4c75-86d5-c71cbc0ba407.png">

Figure 3. Example of Raw Data
 
Class names for new images are correspondence to the online open source. Images are uploaded and labeled [Figure 4].

 <img width="360" alt="image" src="https://user-images.githubusercontent.com/99308255/163247669-7ba6c936-267d-4bb1-9c9f-0117d4ca44f8.png">

Figure 4. Sample from Team’s Data

 <img width="360" alt="image" src="https://user-images.githubusercontent.com/99308255/163247684-94bc05e6-bacb-4f46-85cc-d4a045430f88.png">

Figure 5. Correctly Labeled New Data
![image](https://user-images.githubusercontent.com/99308255/163247573-12ef795d-151c-4658-a6ac-81902ab33f12.png)

### Architecture![image](https://user-images.githubusercontent.com/99308255/163247701-dd708d21-7d19-4235-931d-31e35090d623.png)
<img width="436" alt="image" src="https://user-images.githubusercontent.com/99308255/163247725-f48d59e7-f2e9-4342-b393-1b6150b3c8f0.png">
Figure 6. ResNet Architecture overview

The team attempted VGG, GoogLeNet, and Resnet (regarding shared colab file), but ResNet yields the best accuracy overall. So, our prominent architecture is transfer learning based on ResNet 18. We chose ResNet (Residual Neural Network) because it is advanced from VGG19, our other proposed transfer learning model. ResNet is a CNN layer with 18 layers. Compared to ours with 3 CNN layers baseline, ResNet 18 has deeper layers, thus performing better on complicated tasks. Convolution with stride=2 is utilized for down sampling, and the model used global average pooling to replace our originally designated Fully connected classification layers. Like VGG, a Feature map is controlled such that as the size of the feather map is halved, the number of feature maps will correspond double. This model is well designed to perform various tasks. Moreover, we loaded its pre-trained feature to boost further our project to save time.  
We employed above ResNet as python object from torchvision model=models.resnet.resnet18 with pretrained results imported. The classification layer is still FC like the baseline. It is defined as a function inbuilt in our resnet_model class via model.fc=nn.linear(), with input as the output number of layers from the ResNet and output 43 identical to our number of traffic sign classes. Then we attempt the usage of GPU of Google colab with “cuda:0” as a device to our Resnet model object. Regarding shared colab link for detail
![image](https://user-images.githubusercontent.com/99308255/163247748-bf6d8ac2-47e2-42a7-88f8-807b08f935f5.png)

### Baseline Model ![image](https://user-images.githubusercontent.com/99308255/163247777-bf3c7c6a-e785-449f-ad1c-42072f1bfb03.png)
<img width="451" alt="image" src="https://user-images.githubusercontent.com/99308255/163247797-2ac7a789-2a9e-4293-b056-95fdc4999a11.png">
Figure 7. Baseline Architecture Overview

Our baseline model is a Simple Convolutional Neural Network with ANN Classifiers. This is adapted from in class examples. Proposed baseline model was with 2 convolutional layers + pooling layer, 2 FC layers with Relu applied, and with softmax activation applied. This is proposed to be simple and easy to train. However, seeing that our project’s complicated output (43 outputs after softmax), and the unsatisfactory result yield from original proposed architecture, we added one more convolutional layer and one more Classification ANN Layer, thus achieved a decent accuracy. 
The finalized Baseline CNN model “BASE” is of 3 convolution layers with kernal2 stride2 pooling maxpool2d(2,2) applied after each for feature extraction; this is flattened using view (-1,_) to 512 units, and connected to a FC classification layer with 1 hidden layer with hidden units gradually decreasing at rate of 1/4 per layer progress thus reducing 512 down to 43 units, with softmax applied in crossentropyloss(). Predictions can thence be made. Regard shared colab link for detail.

![image](https://user-images.githubusercontent.com/99308255/163247814-c56da7ca-9f61-4cb1-93e3-f1c8b85dc5df.png)


## Reference
[1] Shanmukh, “Traffic sign recognition cropped images,” Kaggle, 05-Mar-2021. [Online]. Available: https://www.kaggle.com/shanmukh05/traffic-sign-cropped. [Accessed: 02-Mar-2022]. 

## Appendix
