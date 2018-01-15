# Machine Learning Engineer Nanodegree
## Capstone Proposal
Pongrut Palarpong<br/>
January 1st, 2018

## Proposal

### Domain Background

In Thailand, person identification methods have been continuing to evolve due to public security issues. The Office of The National Broadcasting and Telecommunications Commission Thailand (NBTC) has launched the new method of registering a new mobile phone SIM card since December 15, 2017. The biometric verification system this new registration method has been used across Thailand in all 55,000 locations which are mobile phone operators’ customer service shops and the networks of the mobile phone operators’ dealers and partners.

Under this new method, new SIM card subscribers will have insert their ID cards into the finger print reader or the face recognition card reader at the registration locations. The card readers are connected to the mobile phones or PC of the registration locations. In the case of the face recognition system, the locations will take the SIM card users’ face with the mobile phones embedded with an NBTC registration application. Then the app will see if the captured face matches with the face stored in the ID card database. As expected face recognition method is more popular than fingerprint due to the added cost of card readers at service points, which range from 500 baht ($15) to 9,000 baht ($271) but face recognition is no extra cost, it requires only application installed.  Since the biometric verification system has been launched, there are two main problems with the system.
1. The result of the face recognition that the person in the ID card with the photograph is not matched with sim subscriber's photo, despite the fact the same person.
2. There was a problem that the service points agent used photo from the ID card submitted as current person photo due to difficulty from the first problem.

The capstone project will focus on the demonstration of solution idea for the service points agent used photo from the ID card's submitted as current person photo problem. The object detection technique that which is a part of computer vision will be applied as the main solution.
The concept idea is reducing the shooting time from 2 times to 1 time only, which will force agents to capture the ID card with the current photo of SIM subscriber. However, to add more security step, the object detection to verify that the submitted photo has contain the ID card with photo of sim subscriber.


### Problem Statement

Solution to relieve agent used photo from the ID card submitted as current person photo problem. The system will accept only 1 photo which contain both ID card and current photo to create obstacle for fake photo. The challenge of sending a single image is to locate the ID card and the current photo of the SIM subscriber in the image. Therefore, this project needs to simulate ID card detection by locating the ID card from the image.

Due to privacy restrictions, this project uses a dummy employee card instead of a real ID card. The program will be created to receive an image then accurately locate the dummy employee card. However, it is possible that the agent may put the ID card on the prepared photo and then submit it. This extreme case may be detected by anti-spoofing liveness detection mechanism e.g. eye blinking detection or pupil dilation response, but anti-spoofing liveness detection is not present in scope of this project.

The development pipelines are as follows:
1. Creating 1,030 dummy employee's ID cards by using personal faces from the LFW face dataset, and put them on a dummy employee cards.
2. Prepare 1,000 images from the Open Images dataset for use as a background images of train and evaluate datasets.
3. Create 1,000 images dataset for training and evaluating in convolutional neural networks by using 1,000 images of employee's ID cards  and randomly resize,
rotate, adjust image  gamma , and then place randomly on a prepared background.
4. Prepare test dataset by downloading 30 more images of person from google image in order to make it close to real scene photo and then bring the remaining 30 employee's ID cards to randomly placed on the already downloaded image.
5. This step is to label dummy employee's ID card by creating an image annotation of all 1,030 cards on the xml format file.
6. Split 1,000 images dataset to be 900 train dataset, 100 for evaluate dataset and also another real scene 30 for test dataset.
7. Combine all images & labels and then convert into a single tfrecord file for both train and evaluate datasets.
8. Use the test dataset for training in tensorflow object detection api and evaluate dataset used to evaluate mean average precision (mAP).
9. Convert the trained model into a frozen graph consisting of the model architecture and weights in one file in .pb format.
10. Load the trained model in tensorflow object detection api to test on 30 test dataset to present model accuracy.

### Datasets and Inputs

This project uses three types of image datasets: 1.) a person's face, 2.) another image without a Dummy Card in the picture, 3.) person portrait images.
1. Creating a Dummy Card requires a human face image to simulate the replacement of an ID card by using a personal face images from the LFW dataset [2].
There are 13,237 total human images from 5,750 individuals and each image size is 250x250 pixels.
The image files will be shuffled then select images of people that contain more than 1 image in the folder. After that, the image will be filtered out image that contain more than one face because the ID card must have only one face. Finally, select only 1,030 first images to be used for Dummy Card creation.
2. The background images will use images from the Open Images Dataset V3 dataset [3], which provides 41,620 image URLs of validation set.
Shuffle and select for 1,000 images from the original URLs with larger than 768 pixels width and height image size, then resize them with width and length of at least 768 pixels which will maintain the original aspect ratio.
3. Personal portrait images will be simulated for real situation testing by using 30 Dummy Card  and the 30 person names to search and download person images that larger than 1024 pixels from the google images, in order to keep the photos in the Dummy Card which will not be too small that they can not be detected in future facial recognition process.


### Solution Statement

The photo was taken by the agent who photographed the photo of SIM subscriber's ID card and reuse it as current SIM subscriber photo.  This problem occurred with the original biometric verification system, the system receives two images, the ID card and the current photo of the SIM subscriber.
The solution of this problem can be prevented by having the agent photograph the current SIM subscriber photo with his/her ID card in a single photo. Using only one image requires an intermediary program for image separation.  Thus, it need to detect two images and send them to the original system in order to split the image,
there are 2 types of objects must be detected: the current photo of SIM subscriber and  ID card photo.

In the first object type need face detection mechanism, dlib Face Detector will be used as a method for detecting faces because of Face Detector dlib can detect the face effectively [4] .
The second object type is ID card which will be represented by the Dummy Card  The TensorFlow Object Detection API will be used to detect the location of the Dummy Card of the submitted image. Finally, program will display two faces from the current photo of SIM subscriber and the face from the Dummy Card  The detection precision of Dummy Card will be at 0.8 mAP@0.5IOU [7] from the 30 images in test dataset.


### Benchmark Model

This project will use ssd_mobilenet_v1_coco pre-trained detection models [9] as the transfer learning base for the Dummy Card Detection Model.
The performance of ssd_mobilenet_v1_coco is 21 mAP which is the subset of the MSCOCO evaluation protocol. However, the most popular metric uses for evaluating the quality of object detection is Mean Average Precision (mAP) on the protocol of the PASCAL VOC Challenge 2007 . Therefore, this project benchmark model will base on the PASCAL VOC 2007 metrics and to match the benchmark model fairly with benchmark model.

The benchmark model is Dat Tran's Raccoon detector [8] performance at the best score of 0.8 mAP@0.5IOU. The Raccoon detector uses 160 images for training and 40 images for evaluation.  The Dummy Card detector model is expected to score at 0.8 mAP@0.5IOU as same as the Raccoon detector but use 900 images for training and 100 images for evaluating.


### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------
[1] [New SIM registration to require biometric ID starting Dec 15](http://www.nationmultimedia.com/detail/Corporate/30330973)

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
