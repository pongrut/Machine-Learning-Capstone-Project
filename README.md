# Dummy Card Detection Capstone Project

One Paragraph of project description goes here

## Getting Started

In Thailand, person identification methods have been continuing to evolve due to public security issues. The Office of The National Broadcasting and Telecommunications Commission Thailand (NBTC) has launched the new method of registering a new mobile phone SIM card since December 15, 2017. The biometric verification system this new registration method has been used across Thailand in all 55,000 locations which are mobile phone operators’ customer service shops and the networks of the mobile phone operators’ dealers and partners [1].
Since the biometric verification system has been launched, there are two main problems with the system.
1.	Face recognition failure rate of real person with real ID card is still too high.
2.	Agent submits fake photo by photograph person photo from ID card.

This capstone project aims to solve 2nd problem of face recognition system which is "agent submits fake photo". The object detection is a technology related to computer vision, it is applied as the key algorithm for resolve this problem. Preventive policy for fake photo problem is to force agents to submit one photo instead of two photos. Challenge of receiving a single image is that to locate the ID card and then detect person face in submitted image which requires object detection algorithm to complete this task.


### Prerequisites

These softwares and libraries need to bed installed.

```
- Python 3.5
- Anaconda
- Jupyter Notebook
- Scikit Image
- Tensorflow 1.4
- dlib
- Opencv
```

### Installing

The following steps are instructions to get a development env running


```
- conda create -n card-detection -y python=3.5 jupyter nb_conda anaconda scikit-image 
- source activate card-detection
- conda install -c conda-forge tensorflow
- conda install -c conda-forge dlib 
- conda install -c conda-forge ffmpeg
- conda install -c conda-forge/label/broken opencv
```

