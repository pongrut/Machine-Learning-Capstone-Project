# Machine Learning Engineer Nanodegree
## Capstone Project
Pongrut Palarpong  
February 1, 2018

## I. Definition

### Project Overview
In Thailand, person identification methods have been continuing to evolve due to public security issues. The Office of The National Broadcasting and Telecommunications Commission Thailand (NBTC) has launched the new method of registering a new mobile phone SIM card since December 15, 2017. The biometric verification system this new registration method has been used across Thailand in all 55,000 locations which are mobile phone operators’ customer service shops and the networks of the mobile phone operators’ dealers and partners [1].

Under this new method, new SIM card subscribers will have insert their ID cards into the finger print reader or the face recognition at the registration locations. The card readers are connected to the mobile phones or PC of the registration locations. In the case of the face recognition system, the locations will take the SIM card users’ face with the mobile phones embedded with an NBTC registration application. Then the app will see if the captured face matches with the face stored in the ID card database. As expected face recognition method is more popular than fingerprint due to an added cost of card readers at registration service shops which range from 500 baht ($15) to 9,000 baht ($271). Face recognition is no extra cost, it requires only application installed. Since the biometric verification system has been launched, there are two main problems with the system.

1. Face recognition failure rate of real person with real ID card is still too high.
2. Agent submits fake photo by photograph person photo from ID card.


### Problem Statement
This capstone project aims to solve 2nd problem of  face recognition system which is "agent submits fake photo". The object detection is a technology related to computer vision, it was applied as the key algorithm for resolve the problem. Preventive policy for fake photo problem is  to force agents to submit 1 photo instead of 2 photos. Challenge of receiving a single image is that to locate the ID card and then detect person face in submitted image which requires object detection algorithm to complete this task.

However, ID card is not general object that available in pre-trained models in order to make object detection algorithm able to detect ID card requires retrain the pre-trained models. Due to privacy restrictions, this project trains and detects the Dummy Cards instead of a real ID cards. Developed program is a simulation of person verification step start from receiving submitted image from agents and then locate the Dummy Card and also person face. It is possible that the agent may put the ID card on the prepared photo and then submit it. This extreme case may be detected by anti-spoofing liveness detection mechanism e.g. eye blinking detection or pupil dilation response but anti-spoofing liveness detection is not present in scope of this project.

Dummy Card detection uses the TensorFlow Object Detection API [4] as an object detection framework, it  is an open source framework built on top of TensorFlow that makes it easy to construct, train and deploy object detection models. This framework also provides a pre-trained detection set in the COCO data set, the Kitti dataset, and the Open Images dataset. These models can be useful for out-of-the-box in categories already in COCO (eg, human, automobile, etc.).

### Metrics
The evaluation of detection task will be judged by the precision/recall curve in which 2 factors are two main parts: the detection are relevant to the object being detected (precision), and the number of detected objects that are relevant (recall).

**Precision** = [True Positives/(True Positives + False Positives)]<br/>
   **Recall** = [True Positives/(True Positives + False Negatives)]<br/>

- True positives: Dummy Card images that the model correctly detected
- False positives: Not Dummy Card images that the model incorrectly detected, classifying them as Dummy Cards.
- False negatives: Dummy Card images that the model did not detect as Dummy Cards.

The principal quantitative measure used will be the mean average precision (mAP) of PASCAL VOC 2007 metrics. Detections are considered True positives and False positives from the area overlapping with predicted bounding boxes BP and ground truth bounding boxes GTB (Intersection Over Union). Intersection Over Union (IOU) is an evaluation metric used to evaluate the accuracy of object detection model. If BP and GTB overlap 100% is the maximum accuracy. However, the mAP@0.5IOU is determined that overlap between the predicted bounding box of BP and the ground truth bounding box GTB must exceed 50% the prediction is considered a true positive. To calculate mean average precision (mAP) metrics by looping overall all ground-truth bounding boxes in the validation dataset (Fig.1).

![IOU Formula](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/Figure5-624x362.png)<br/>
Figure 1: Illustration of Intersection Over Union (IOU) Calculation.


## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
