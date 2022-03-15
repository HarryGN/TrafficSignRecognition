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
<p align="center">
<img width="363" alt="Screen Shot 2022-03-02 at 8 12 34 PM" src="https://user-images.githubusercontent.com/99308255/156477015-48825a7b-f4be-468e-b27e-733f28c68e8b.png">
</p>

### Collection
Datasets for ML training purposes are very easy to find nowadays as people are aware of the importance of having clear and realistic data. Some authenticated sources are data.gov.in, Kaggle, and UCI dataset repository. The best study material can let the model learn to obtain the best test results.
### Preparation
The dataset we are using is from kaggle.com. Traffic sign recognition cropped images. This dataset contains the cropped images of original datasets by GTSRB in jpg format. There are over 27000 training images and 12630 testing images. 
<img align="left" width="360" height="282" src="https://user-images.githubusercontent.com/99308255/156477234-a2bc53cc-84bc-44ad-817a-c09a3d1605df.png">

<img align="right" width="340" height="282" src="https://user-images.githubusercontent.com/99308255/156477264-86f11a87-8cf8-46cc-8687-109a8d340d7b.png"> 
<br clear="right"/>
Many users have verified this dataset at Kaggle,, and 100% of them are labeled valid by many users. [Figure 2] Here is a preview of these test data. This dataset is very realistic because they are taken at different periods and weathers,, and the lighten is real. [Figure 3]

## Reference
[1] Shanmukh, “Traffic sign recognition cropped images,” Kaggle, 05-Mar-2021. [Online]. Available: https://www.kaggle.com/shanmukh05/traffic-sign-cropped. [Accessed: 02-Mar-2022]. 

## Appendix
### Updated Project Plan (Appendix A)
**Data Processing(Harry)** including Future Plan:

The team will progress with the automatic cropping (and potential labeling) of DVR images derived by Harry. Proposed technique to be applied: “Instance segmentation learning”, however, the team will consult with TA to verify for feasibility and attain recommendation of the best-suited application for data processing for our project, inquiry currently proposed for research and TA question session:
Would it be possible to train the model with more examples that have already been cropped and use the image labeled and cropped by the team as the validation and test data?

After receiving answers to these questions, the team will use readily available datasets and apply to resize and any other operation applicable before feeding these data into the model for training/validation/testing purposes. 
Cropping images from the DVR provided by Harry and label them so that the team can have an authentic source for validation and testing.

The team aims to simultaneously develop the network with its architecture and data processing. However, data processing is the most critical step in this model training process. It needs to be dealt with seriously to make sure the images and labels we use should be as authentic and valid as possible. Therefore, more datasets should be used to train and valia& test with our own data.

 The data processing step should be finished by **March 2nd**. this is the primary deadline we set for the project proposal. Since we already have the data we need for training, we can crop and resize the images from the DVR later and the team is proposing to finish the clean by**March 10th.**

**Baseline Model(Mary) & Primary Model Classification layer(Lurui)** including future plan:

Baseline Model is done mainly by Mary and the proposed baseline model was the SVM model, as it maintains a high accuracy but we can still try to improve on that as we gradually updating our knowledge:
Adapt some tuning to the basic CNN model, modify the number of layers and get the training accuracy result. We can reduce the time consumed because we choose to use numbered training examples with the usage of GPU . this process will be done by Mar. 7th as we already have the train data.
Resize the images from data processing again as SVM can only receive input of the same size. (224 x 224 pixels) **[Deadline: Mar. 10th]** Make sure that the input for the baseline model are the same for the model we are going to develop.
After the final decision of the choice is made, the team needs to modify the overall documentation and include justifications for the choice. **[Deadline: Mar. 20th]**

**nThe classification layer is part of the primary model and is responsible (Lurui)**
Classification layer in Primary model well optimized for the given 1x1000 linear output. This process may vary as we are not that that stage yet, the choice may differ.

**Primary Model Feature recognition layer(Zidong)**

The abnormally high result accuracy yielded from RESNET may be due to an error in the training function. Will review training function and dataset extraction program. **[deadline: Mar. 7th]**
Attempt on different Transfer Learning model, MobileNet was proposed by Harry, should learn how to implement and attempt to compare results (only after 1 is completed). **[deadline: March 14th]**
Will proceed to tweak on different hyperparameters to yield better results. **[deadline: TBA, this is a continuous process and will change as the network is updated]**
Will conduct a question session with Class TA to ask proposed questions 1) about Data Processing and 2) about Feature Recognition layer as the question arises. **[deadline: March 15th, during PRA session]**

