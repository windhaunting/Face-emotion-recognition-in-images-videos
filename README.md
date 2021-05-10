

# Face emotion recognition in images and videos

We explore human recognition system to identify 7 types of emotions by using FER2013 dataset as a pretrained model features.
Furthermor, we apply this CNN pretrained model features to a Long Short-Term Memory(LSTM) networks and using two
datasets(JAFFE and AFEW) to train it and evaluate it. Finally, we use this
model to test on image datasets and video datasets to compare the performance and get relative high accuracy with and without CNN pretrained model features.

## Table of content

- [Installation](#installation)
- [Datasets](#datasets)
- [Model flow](#model-flow)

## Installation  

- python 3.3 or more
- opencv 3.4 or more  
- Keras
- TFLearn


## Datasets

used for training and testing:

- JAFFE 
- CK+2
- Wild (AFEW) 5.0


# Model flow

The model architecture is as follow:

![model architecture](https://github.com/windhaunting/face-emotion-recognition-in-images-videos/blob/master/model_flow.png)

