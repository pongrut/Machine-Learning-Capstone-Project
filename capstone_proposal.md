# Machine Learning Engineer Nanodegree
## Capstone Proposal
Pongrut Palarpong<br/>
January 1st, 2018

## Proposal

### Domain Background

In Thailand, person identification methods have been continuing to evolve due to public security issues. The Office of The National Broadcasting and Telecommunications Commission Thailand ([NBTC](https://www.nbtc.go.th/About/history3.aspx)) has launched the new method of registering a new mobile phone SIM card since December 15, 2017. The biometric verification system this new registration method has been used across Thailand in all 55,000 locations which are mobile phone operators’ customer service shops and the networks of the mobile phone operators’ dealers and partners[[1](http://www.nationmultimedia.com/detail/Corporate/30330973)].

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
_(approx. 2-3 paragraphs)_

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

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
