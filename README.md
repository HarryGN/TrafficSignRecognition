# TrafficSignRecognition
Traffic Sign Recognition Using Deep Learning on Self-Driving Cars
## Introduction
Self-driving cars are autonomous decision-making systems. They process data from sensors such 
as cameras, LiDAR, RADAR, GPS, or inertia sensors. This data is then modeled using deep 
learning algorithms, making decisions relevant to the car's environment. [1] This project
investigates how image recognition is used in self-driving cars. How deep learning, especially 
convolutional neural networks, models traffic sign recognition. In this project, we are going to 
focus more on the image taken by the camera therefore computer vision is crucial. Thus, we are 
taking CNNs as our prior choice. The traffic dataset we will be using is the Canadian traffic sign 
data set.

Interesting things happen when the team realizes that noise in the images is inevitable. [Figure 1: 
Example of noise]. Image filtering is then an add-on to this traffic sign recognition feature. In 
some extreme weather and corner cases, noise would cause a certain wrong output of the 
algorithm and the team’s goal is to propose an algorithm composed of three processing stages: 
detection, pictogram extraction, image filtering and classification. [2]

![01BB36F0-5B7C-4F16-9BEF-86B02B7B4D68](https://user-images.githubusercontent.com/99308255/153995403-1fb21713-3ac4-4ccd-b794-7f66baefa553.jpeg)

This is an overall though map of how this CNN is implemented when spatial transformers have 
been deployed -> determine noise (unique feature here) and do image filtering -> uses CNN for 
extrusion and then uses a linear classifier [3]
## Data Processing
Data processing is the step I am responsible for and is very tightly connected with things my colleagues are about to do. I am taking this step very seriously as this is a deductive process for this traffic sign recognition network training. To be as realistic as possible, the team has agreed to use videos recorded from the DVRs. However, the data we need is only the traffic sign part. Suppose we insist on doing that, we will need to label every information and classify them. Therefore, it would still not be the best approach. 

Here are two reasons. First is the significant limitation. The place that this vehicle has been to is minimal and would possibly give highly repeated images from the same perspective, and this is not ideal for a model to train. Second, the processing effort is possible. According to an open-source GitHub process, we need to extract frames and filter those images that do not include traffic signs. This step is taking too long and not ethical, considering we have a network to build afterward. Thus, I choose to look for pre-labeled data about traffic signs for ML training purposes and choose some of the pictures from DVRs to validate and test the model to match our initial objective of being realistic.
