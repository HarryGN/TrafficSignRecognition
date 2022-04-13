# Traffic Sign Recognition


## 1. Introduction
Self-driving cars are autonomous decision-making systems. They process data from sensors such as cameras, LiDAR, RADAR, GPS, or inertia sensors [1], and the data is then loaded to internal systems where predictions are made. Then, the car can react accordingly to the street conditions identified. Self-driving cars are driving much more attention than before, as it is an energy-efficient and labor-saving approach to transportation [2].

This project investigates how image recognition is used in self-driving cars, which serves to be the first step towards self-driving. The resultant deep learning model will make predictions to the traffic sign images input and categorize them into correct classes. Thus, the predictions could be used for further mechatronic design of the driving system of the car. 

Deep CNN is a suitable tool to achieve our goal, as the algorithms allow the automatic detection of desired objects and has excellent functionality in categorizing common features in “similar” images, observing trends from mass input, and predicting the result into multiple classes. Thus, the system could “learn” better after each training process and as the number of samples increased. Figure 1 shows the general process plan of the project.

<p align="center">
<img width="452" alt="image" src="https://user-images.githubusercontent.com/99308255/163248262-676ad70e-452b-4854-97bb-f6ebd44d6308.png">
</p>
<p align="center">
Figure 1. Process illustration
</p>

## 2. Background & Related Works
Many researchers have determined traffic sign recognition as the initial step for this process. This technique is generally divided into two parts, one is the detection which involves the use of Yolo, the other is the classification in a research paper conducted 2019 [3]. We are borrowing the same thought map and focusing on the development of traffic sign classification in deep CNN. The image segmentation and detection are done using an open-source code using YOLOv5from GitHub to make our piece complete and more useful in a general form [4]. YoloV5 is introduced in a research paper used in segmentation step [5] and traffic signs are cropped out for our model testing purposes. Since the requirement for training the dataset needs to be large and accurate. The training and validation dataset is obtained from CTS (Chinese Traffic Signs) [6].

## 3. Data Processing 
The data processing of the project consists of two steps: the first stage was to adapt an external online dataset for training and the second was to generate and collect our own dataset. The online source will serve as the training set and our own generated data will be used as the validation and testing set.

During the first stage, the team obtained data from the Chinese Traffic Sign Database [6], which was supported by National Nature Science Foundation of China. The database consists of accurately cropped and labeled data where the classes are represented by numbers to increase efficiency. Figure 2 shows some of the sample data.

<p align="center">
<img width="252" alt="image" src="https://user-images.githubusercontent.com/99308255/163248531-b338214b-4ae6-4166-8a67-d11e9d65c79e.png">
</p>
<p align="center">
Figure 2. Kaggle Sample Data
</p>

To generate data by us, the team collected the street images in front of the university using phone camera and manually crop them into desired size. Image segmentation technique was applied to do the automatic detection and cropping. Figure 3 shows an example of the raw data. 

<p align="center">
<img width="135" alt="image" src="https://user-images.githubusercontent.com/99308255/163248595-10266b7b-d520-4d1d-9028-77e7e11fe257.png">
</p>
 
<p align="center">
Figure 3. Example of Raw Data
</p>
Class names for new images are correspondence to the online open source. Images are uploaded and labeled [Figure 4].

<p align="center">
<img width="360" alt="image" src="https://user-images.githubusercontent.com/99308255/163248618-faf6943d-70f6-4464-834d-f8992c237297.png"> 
</p>
<p align="center">
Figure 4. Sample from Team’s Data
</p>
 
<p align="center">
<img width="360" alt="image" src="https://user-images.githubusercontent.com/99308255/163248635-7aacb0de-cfa8-42a6-8dab-42017e4d62a1.png">
</p>
<p align="center">
Figure 5. Correctly Labeled New Data
</p>


## 4. Architecture

<p align="center">
<img width="436" alt="image" src="https://user-images.githubusercontent.com/99308255/163248650-8264779a-fc9f-4c94-a05a-961f0d508c7d.png">
</p>
<p align="center">
Figure 6. ResNet Architecture overview
</p>

The team attempted VGG, GoogLeNet, and Resnet (regarding shared colab file), but ResNet yields the best accuracy overall. So, our prominent architecture is transfer learning based on ResNet 18. We chose ResNet (Residual Neural Network) because it is advanced from VGG19, our other proposed transfer learning model. ResNet is a CNN layer with 18 layers. Compared to ours with 3 CNN layers baseline, ResNet 18 has deeper layers, thus performing better on complicated tasks. Convolution with stride=2 is utilized for down sampling, and the model used global average pooling to replace our originally designated Fully connected classification layers. Like VGG, a Feature map is controlled such that as the size of the feather map is halved, the number of feature maps will correspond double. This model is well designed to perform various tasks. Moreover, we loaded its pre-trained feature to boost further our project to save time.  

We employed above ResNet as python object from torchvision model=models.resnet.resnet18 with pretrained results imported. The classification layer is still FC like the baseline. It is defined as a function inbuilt in our resnet_model class via model.fc=nn.linear(), with input as the output number of layers from the ResNet and output 43 identical to our number of traffic sign classes. Then we attempt the usage of GPU of Google colab with “cuda:0” as a device to our Resnet model object. Regarding shared colab link for detail

## 5. Baseline Model 

<p align="center">
<img width="451" alt="image" src="https://user-images.githubusercontent.com/99308255/163248687-c5cf887c-b7ab-43a3-9ad7-030ea923f290.png">
</p>
<p align="center"> 
Figure 7. Baseline Architecture Overview
</p>

Our baseline model is a Simple Convolutional Neural Network with ANN Classifiers. This is adapted from in class examples. Proposed baseline model was with 2 convolutional layers + pooling layer, 2 FC layers with Relu applied, and with softmax activation applied. This is proposed to be simple and easy to train. However, seeing that our project’s complicated output (43 outputs after softmax), and the unsatisfactory result yield from original proposed architecture, we added one more convolutional layer and one more Classification ANN Layer, thus achieved a decent accuracy. 
The finalized Baseline CNN model “BASE” is of 3 convolution layers with kernal2 stride2 pooling maxpool2d(2,2) applied after each for feature extraction; this is flattened using view (-1,_) to 512 units, and connected to a FC classification layer with 1 hidden layer with hidden units gradually decreasing at rate of 1/4 per layer progress thus reducing 512 down to 43 units, with softmax applied in crossentropyloss(). Predictions can thence be made. Regard shared colab link for detail.

## 6. Results

To evaluate our model, the quantitative measures we took were: 
1. Compare the accuracy difference between the improved transfer learnings and the baseline model to see the extent of improvements.
2. Compare the training accuracy with the testing accuracy of the same improved model to see whether the model overfits.
3. Feed in the new unseen data (traffic sign images generated by us) and output the results. This will be introduced in detail in section 7.
 
6.1 Accuracy of the Baseline Model
Figure 8 shows the output of the baseline model, where the training accuracy reached 6.195% and the validation accuracy got 4.323%. The final test accuracy was 5.451%, urged the improved model to be applied.

<p align="center"> 
<img width="190" alt="image" src="https://user-images.githubusercontent.com/99308255/163248728-ace77a08-5ab4-41ac-8119-066fac26f83d.png">
</p>
<p align="center"> 
Figure 8. Results for the Baseline Model
</p>

6.2 Accuracy of the Transfer Learning
When the transfer learnings are applied, the overall accuracy gets greatly improved. The team used the GoogLeNet and ResNet to be the final model and both reached an accuracy above 90%, a significant improvement from the baseline model. That is, the model could accurately categorize the class of each traffic sign images with more than 90% of success possibility. This serves to be the first step towards automatic self-driving, where the vehicle could recognize the traffic sign images from the camera input and make accurate predictions on the actual road conditions.


<p align="center">
<img width="133" alt="image" src="https://user-images.githubusercontent.com/99308255/163248764-802c84c9-1fee-40cb-9b29-e655da9dc2b8.png">
</p>
<p align="center"> 
(a) 
</p>
<p align="center"> 
<img width="140" alt="image" src="https://user-images.githubusercontent.com/99308255/163248780-928f5ae5-c0ef-4b6f-a051-94fbb730a455.png">
</p>
<p align="center"> 
(b)  
</p>
<p align="center"> 
<img width="135" alt="image" src="https://user-images.githubusercontent.com/99308255/163248835-56d1f391-ccc1-44bd-a324-01bc5ee076a6.png">
</p>
<p align="center"> 
(c)
</p>
<p align="center"> 
Figure 9. History best results for (a) VGG; (b) GoogLeNet; (c) ResNet
</p>

From the results for those two transfer learnings, we observe that ResNet model yields a higher validation accuracy and the gap between the training and validation gets smaller as the number of iterations increases. Thus, ResNet is selected to be the final suitable model for our specific application.

However, there still exist some limitations to the model. First, the data input is constrained to be the 43 classes the team selected. Other traffic signs, besides the identified ones, will be neglected during the recognition process. Second, if the traffic signs from the related official offices are changed, the classes need to be relabeled. Third, the naming of traffic signs will differ according to the region, policy, and religion. 

## 7. Evaluation of the Model on the New Data
The new data was obtained from traffic signs outside of the university. We also made some pre-processes for the testing convivence. We are aiming to test the model build for cropped traffic sign recognition, not the image segmentation. Therefore, traffic signs will then be cropped out first. We used iPhone camera when processing new data to evaluate the model. That been said these three images we chose have never been seen by the model and the output is purely based on the trained function to predict the result. The labels are put into a class which will output at the top of the output. Aside outputting a prediction, we also give the confidence of it in an array which is guided by equation:

<p align="center"> 
𝑐𝑜𝑛𝑓𝑖𝑑𝑒𝑛𝑐𝑒 = 𝑚𝑎𝑥(𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛))
</p>
 


This is basically the probability of which of the class the image should be classified. Some output is show in the figure below. 
<p align="center"> 
<img width="451" alt="image" src="https://user-images.githubusercontent.com/99308255/163253923-c788e508-5195-4003-a2b5-bbfbf060f1d4.png">
</p>
<p align="center"> 
(a)
</p>
<p align="center"> 
<img width="451" alt="image" src="https://user-images.githubusercontent.com/99308255/163253172-1f18ff20-eefa-43ce-8273-bc5e67f3ed41.png">
</p>
<p align="center"> 
(b)
 </p>
<p align="center"> 
<img width="451" alt="image" src="https://user-images.githubusercontent.com/99308255/163253211-3f20f293-a088-4d81-864e-523624a713bc.png">
</p>
<p align="center"> 
(c)
 </p>
<p align="center"> 
Figure 10: Model Prediction for (a) Baseline; (b) GoogLeNet; (c)ResNet
</p>


Although the baseline model output the classification correctly, the confidence of which the image’s probability been classified into the class is low. The team has agreed that the correctness of the baseline model is purely by luck as demonstrated earlier that the training accuracy for the baseline model is ridiculously low. However, the confidence for the prediction in GoogLeNet and ResNet is very high, especially ResNet, which is obtaining 99.99% confidence for the first image, 99.94% for the second image, 100.00% for the third. From this comparison, the team has agreed that ResNet is the most reliable and precise in this demonstration, where the baseline model is the least.

This can be proved by the accuracy obtained in previous part, baseline model only obtained a 5% accuracy during training and validation, where ResNet has a 98.8% training accuracy and a 97.5% validation accuracy. Since the validation dataset has 532 images that is new to the model. Comparing the training accuracy curve and the validation curve there is neither over or underfitting issues nor any bias-variance trade-off. The validation accuracy curve also is getting converged which shows a good adjustment of the hyperparameters.

## 8. Discussion
Our project's input images are of varied resolutions and are required for numerous outputs (43 outputs for 43 classes), so the number of parameters must be massive, and the task is implicitly complicated. This requires a deeper learning network than our proposed Baseline CNN. For this need for massive parameter computation and deeper layers, we found ResNet to be optimal since it is deep and is with stride number 2 between convolutions, which effectively prevents gradient vanishing and allows the model to learn deeper features rather than superficial features. Due to this advantage this model outperforms VGG16, which have similarly deep layers but do not have convolution stride. We employed VGG16 and ResNet18 in parallel to keep several layers to discuss this discrepancy. VGG16 has a dilutionary result of below 10% training and testing accuracy; this is caused due to the vanish of gradient due to deep layers and massive parameter as discussed. ResNet18 has similar layer construction, but it yields almost perfect results. NOTES: the data source and number of epochs are same for each model, so it is reasonable to make comparisons between those.

For the baseline model, the final training accuracy and the validation accuracy are 0.06195 and 0.04323, respectively. The low train accuracy implies the baseline model is not able to become overfitting. Moreover, the model that cannot overfit is generally weaker. However, the small gap between its validation accuracy and train accuracy also shows that the model is highly generalized, and another advantage of the baseline model is that it only costs 7 minutes to finish the training, five epochs for each model, which is the fast model in our projects.

After the baseline model, the model with the transfer learning technique will be discussed. The first post train network our team selected is the GoogLeNet. Therefore, the discussion will also start from this model. The final training accuracy and validation accuracy are 0.9787 and 0.91917, respectively. The high training accuracy and a more significant gap between the training accuracy and the validation accuracy imply that the model can overfit the training data and has a high potential to find a suitable solution. This point is also illustrated by its high accuracy in both validation and training. The training time cost to train GoogLeNet is only 10 minutes which is only 3 minutes more than the baseline model, but the accuracy rate of the GoogLeNet is almost 20 times higher than the baseline model.

The next model discussed is the VGG16 model, the worst model among the three-transfer learning models. The final validation accuracy and train accuracy are 0.04693 and 0.0582, respectively, which is even lower than the baseline model's accuracy rate, which means this model is even weaker than our baseline line model. What is more, it took almost 40 minutes to train it.

The last model discussed would be the Resnet18. This model yields the highest training and validation accuracy rate among those models, with 0.988 and 0.9755, respectively. The small gap between these two-accuracy rates illustrates the generalization of this model, and high training accuracy also illustrates that the model is strong and able to remember all the features for different classifications. However, this model costs a little bit longer than GoogLeNet to train, taking around 30 minutes. 

In conclusion, with the 5-epoch limit due to limited time conditions, Reset's performance curve shows that no significant overfitting is present, Google Net is with some overfitting, and VGG is with significant overfitting. Google Net can outperform VGG due to its ability to utilize Inception modules to dynamically choose different filter sizes for optimized feature map generation and reduce the number of parameters, compared to VGG, which only offers vanilla convolutional computations with fixed filter size. Though that being the case, Google Net has 22 layers, which is still more layers than ResNet18, which, seeing its results, we perceive as enough layers. GoogLeNet with more layers could be redundant. It, therefore, has a chance to learn unnecessarily in-depth features from our traffic sign samples and thus has overfitted on high accuracy range.

## 9. Ethical Considerations
Almost training databases and the validation database are the traffic signs from Canada. Therefore, the model is expected to predominately learn about the traffic sign from Canada.

If this model is applied to a country other than Canada and is used to detect the traffic sign beside Canada, it is highly possible to determine the sign incorrectly and cause serious car accidents. The accidents which only happen beside Canada may cause a serious bias against the other country such as China or Africa, because this model can work correctly in Canada, but work incorrectly in their countries.



 
## 10. Reference
[1] Wikipedia, Dashcam, [Online]. Available at:
      	https://en.wikipedia.org/wiki/Dashcam [Accessed: 13-Apr-2022]
[2] Mindy Support, “How Machine Learning in Automotive Makes Self-Driving Cars a Reality”, [Online]. Available at: https://mindy-support.com/news-post/how-machine-learning-in-automotive-makes-self-driving-cars-a-reality/#:~:text=Machine%20learning%20algorithms%20make%20it,or%20even%20better%20than)%20humans [Accessed: 13-Apr-2022]
[3] D. A. Alghmghama, G. Latif, J. Alghazo, and L. Alzubaidi, “Autonomous Traffic Sign (ATSR) Detection and Recognition using Deep CNN,” 2019. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1877050919321477. [Accessed: 13-Apr-2022].
[4] Balakishan77, “Balakishan77/yolov5_custom_trained_traffic_sign_detector: Yolov5 Object Detection on traffic signs dataset with custom training,” GitHub. [Online]. Available: https://github.com/Balakishan77/yolov5_custom_trained_traffic_sign_detector. [Accessed: 13-Apr-2022].
[5] Z. Qin and W. Q. Yan, “Traffic-sign recognition using Deep Learning,” SpringerLink, 01-Jan-1970. [Online]. Available: https://link.springer.com/chapter/10.1007/978-3-030-72073-5_2. [Accessed: 13-Apr-2022].
[6] D. Yemelyanov, “Chinese traffic signs,” Kaggle, 16-Apr-2020. [Online]. Available: https://www.kaggle.com/datasets/dmitryyemelyanov/chinese-traffic-signs. [Accessed: 13-Apr-2022].


