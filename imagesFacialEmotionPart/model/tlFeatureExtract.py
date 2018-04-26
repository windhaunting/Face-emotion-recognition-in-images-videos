#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:13:10 2018

@author: fubao
"""
#load the pretrained model weights ; transferred learning features


import numpy as np

import feature_utility as fu
import cv2


from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import modifiedVGG



def modifiedVGG16Model(weights_path=None, shape=(1, 48, 48)):
    '''
    no output softmax
    '''
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=shape))
    model.add(Convolution2D(32, 3, 3, activation='relu'))           # 1
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))           # 3
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))           # 6
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))          # 8
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))         # 11
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))         # 13
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))         # 15
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))                      # 18
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))                       # 20
    model.add(Dropout(0.5))
    
    return model

def extractModifiedVGG(inputImage = None, shape = None):
    
    '''
    get loaded weight from previous model;
    not use the last layer (softmax)
    '''
    weights_path = "my_model_weights_83.h5"
    
    model = modifiedVGG.VGG_16(weights_path=weights_path, shape=(1, 48, 48))
    #print ("model summary: ", model.summary())
    #print ("model weight: ", model.weights, len(model.weights))
    
    #weights = model.layers[20].get_weights()
    #print ("first layer weight: ",len(weights), np.asarray(weights).shape)
    modelNew = modifiedVGG16Model()
    modelNew.set_weights(model.get_weights()[:-1])
    #model.set_weights(model.get_weights())
    print ("modelNew summary: ", modelNew.summary())

    weights = modelNew.layers[20].get_weights()
    print ("firsttt layer weight: ",len(weights), np.asarray(weights).shape)
    
    if inputImage is not None:
        img = fu.preprocessing(cv2.imread(inputImage))
        X = np.expand_dims(img, axis=0)
        X = np.expand_dims(X, axis=0)
        print ("XX: ", X.shape)
            
        featuredX = modelNew.predict(X)
        
        print ("featuredX: ", featuredX.shape)
     
    
if __name__ == "__main__":
    testImage = "../dataSet/jaffe/KA.AN3.41.tiff"
    extractModifiedVGG(inputImage = testImage)


