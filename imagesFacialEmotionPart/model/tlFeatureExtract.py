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

from keras.utils import plot_model
import modifiedVGG

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization


def modifiedVGG16Model(weights_path=None, shape=(1, 256, 256)):
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
    
    print ("modifiedVGG16Model Create model successfully")

    return model

def VGG16Model():
    '''
    use pretrained VGG16 model
    '''
    
    #load vgg16 without dense layer and with theano dim ordering
    base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (3, 256, 256))

    #number of classes in your dataset e.g. 20    
    x = Flatten()(base_model.output)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    #predictions = Dense(7, activation = 'softmax')(x)


    #Add a layer where input is the output of the  second last layer 
    #x = Dense(7, activation='softmax', name='predictions')(vgg16.layers[-2].output)
    
    #Then create the corresponding model 
    newModel = Model(input=base_model.input, output=x)
    newModel.summary()

    return newModel

def extractModifiedVGG16toArray(x, shape):
    
    '''
    get features from an  numpy array sets from the cnn pretrained model
    '''
    
    modelNew = VGG16Model()
    if x is not None:
        cnnfeaturesX = modelNew.predict(x)
        return cnnfeaturesX

def extractModifiedVGGSingleImages(inputImage, shape):
    
    '''
    get loaded weight from previous model;
    not use the last layer (softmax)
    apply on the single image to get features
    '''
    weights_path = "my_model_weights_83.h5"
    
    model = modifiedVGG.VGG_16(weights_path=weights_path, shape=shape)
    #print ("model summary: ", model.summary())
    #print ("model weight: ", model.weights, len(model.weights))
    
    #weights = model.layers[20].get_weights()
    #print ("first layer weight: ",len(weights), np.asarray(weights).shape)
    modelNew = modifiedVGG16Model(weights_path=None, shape=shape)
    modelNew.set_weights(model.get_weights()[:-1])
    #model.set_weights(model.get_weights())
    print ("modelNew summary: ", modelNew.summary())

    weights = modelNew.layers[-2].get_weights()
    print ("firsttt layer weight: ",len(weights), np.asarray(weights).shape)
    
    if inputImage is not None:
        img = fu.preprocessing(cv2.imread(inputImage))
        X = np.expand_dims(img, axis=0)
        X = np.expand_dims(X, axis=0)
        print ("XX: ", X.shape)
            
        featuredX = modelNew.predict(X)
        
        print ("featuredX: ", featuredX.shape)
     

def extractModifiedVGGArray(x, shape):
    
    '''
    get features from an  numpy array sets from the cnn pretrained model
    '''
    
    print ("extractModifiedVGGArray x  shape", x.shape, shape)
    
    weights_path = "my_model_weights_83.h5"
    
    model = modifiedVGG.VGG_16(weights_path=weights_path, shape=shape)
    #print ("model summary: ", model.summary())
    #print ("model weight: ", model.weights, len(model.weights))
    
    #weights = model.layers[-2].get_weights()       # [20]
    #print ("first layer weight: ",len(weights), np.asarray(weights).shape)
    modelNew = modifiedVGG16Model(weights_path = None, shape=shape)
    
    modelNew.set_weights(model.get_weights()[:-1])
    #model.set_weights(model.get_weights())
    print ("modelNew summary: ", modelNew.summary())
    
    if x is not None:
        cnnfeaturesX = modelNew.predict(x)
        return cnnfeaturesX
    
if __name__ == "__main__":
    testImage = "../dataSet/jaffe/Angry/KA.AN3.41.tiff"
    #extractModifiedVGGSingleImages(inputImage = testImage, shape=(1, 256,256))

     #model = modifiedVGG16Model(weights_path=None, shape=(1, 256, 256))
    model = VGG16Model()
    img = image.load_img(testImage, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    cnnfeaturesX = model.predict(x)
    print ("shape: ",cnnfeaturesX.shape, cnnfeaturesX)