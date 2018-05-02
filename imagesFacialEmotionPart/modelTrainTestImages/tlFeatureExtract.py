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
from matplotlib import pyplot as plt


from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.utils import plot_model
from keras import backend as K

import modifiedVGG

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
from CNNActivation import get_activations, display_activations

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

def preprocessing(img, size=(256, 256)):
    #print ("preprocessing img file", img)

    img = cv2.imread(img, 0)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size).astype(np.float32)    
    #img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    #print ("img size", img.shape)
    return img


def modifiedVGG16Model(weights_path=None, shape=(256, 256)):
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

    '''
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))                      # 18
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))                       # 20
    model.add(Dropout(0.5))
    '''
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
        img = preprocessing(inputImage)
        x = np.expand_dims(img, axis=0)
        #X = np.expand_dims(X, axis=0)
        print ("XX: ", x.shape)
            
        x = np.asarray(x)
        # do mean and standardlization 
        x -= np.mean(x, axis=0)
        x /= np.std(x, axis=0)
        print ("x: ", x.shape)
        #featuredX = modelNew.predict(x)
        

    
def extractModifiedVGGArray(x, shape):
    
    '''
    get features from an  numpy array sets from the cnn pretrained model vGG16
    '''
    
    print ("extractModifiedVGGArray x  shape", x.shape, shape)
    
    weights_path = "my_model_weights_83.h5"
    
    model = modifiedVGG.VGG_16(weights_path=weights_path, shape=shape)
    #print ("model summary: ", model.summary())
    #print ("model weight: ", model.weights, len(model.weights))
    
    #weights = model.layers[-2].get_weights()       # [20]
    #print ("first layer weight: ",len(weights), np.asarray(weights).shape)
    modelNew = modifiedVGG16Model(weights_path = None, shape=shape)
    
    modelNew.set_weights(model.get_weights()[:-5])
    #model.set_weights(model.get_weights())
    print ("modelNew summary: ", modelNew.summary())
    
    if x is not None:
        cnnfeaturesX = modelNew.predict(x)
        return cnnfeaturesX
    return None

    
def layer_to_visualize(img_to_visualize, model, layer):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])
    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)

    print ('Shape of conv:', convolutions.shape)
    
    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))
    
    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(12,8))
    for i in range(len(convolutions)):
        ax = fig.add_subplot(n,n,i+1)
        ax.imshow(convolutions[i], cmap='gray')
        
def visualLayersOutput(testImage):
    
        
    img = preprocessing(testImage, size=(256, 256))
    x = np.expand_dims(img, axis=0)
    x = np.asarray(x)

    print (" visualLayersOutput, x shape: ", x.shape)
    
    weights_path = "my_model_weights_83.h5"
    model = modifiedVGG.VGG_16(weights_path=weights_path, shape=(1,256,256))
    #print ("model summary: ", model.summary())
    #print ("model weight: ", model.weights, len(model.weights))
    
    #weights = model.layers[20].get_weights()
    #print ("first layer weight: ",len(weights), np.asarray(weights).shape)
    modelNew = modifiedVGG16Model(weights_path=None, shape=(1, 256,256))
    modelNew.set_weights(model.get_weights()[:-1])
    
    print (" visualLayersOutput, modelNew: ")
    
    a = get_activations(modelNew, x, print_shape_only=True)  # with just one sample.
    display_activations(a)
    
    
    '''
    weights_path = "my_model_weights_83.h5"
    model = modifiedVGG.VGG_16(weights_path=weights_path, shape=(1, 256,256))
    #print ("model summary: ", model.summary())
    #print ("model weight: ", model.weights, len(model.weights))
    
    #weights = model.layers[20].get_weights()
    #print ("first layer weight: ",len(weights), np.asarray(weights).shape)
    modelNew = modifiedVGG16Model(weights_path=None, shape=(1, 256,256))
    modelNew.set_weights(model.get_weights()[:-1])
    
    
    img = preprocessing(testImage, size=(256, 256))
    x = np.expand_dims(img, axis=0)
    #x = np.expand_dims(x, axis=0)
    x = np.asarray(x)

    print (" visualLayersOutput, x shape: ", x.shape)
    
    desiredLayers = [1,5]
    desiredOutputs = [modelNew.layers[i].output for i in desiredLayers] 

    #alternatively, you can use cnnModel.get_layer('layername').output for that    

    newModel = Model(modelNew.inputs, desiredOutputs)
    output = newModel.predict(x)
    print("newModel.predict(x): ", output.shape)

    plt.imshow(output)
    '''
    
def textOnImage(image):
    font = ImageFont.truetype(image,25)
    img=Image.new("RGBA", (200,200),(120,20,20))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0),"This is a test",(255,255,0),font=font)
    draw = ImageDraw.Draw(img)
    draw = ImageDraw.Draw(img)
    img.save("a_test.png")

if __name__ == "__main__":
    '''
    testImage = "../dataSet/jaffe/Angry/KA.AN3.41.tiff"
    #extractModifiedVGGSingleImages(inputImage = testImage, shape=(1, 256,256))

     #model = modifiedVGG16Model(weights_path=None, shape=(1, 256, 256))
    model = VGG16Model()
    img = image.load_img(testImage, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    cnnfeaturesX = model.predict(x)
    print ("shape: ",cnnfeaturesX.shape, cnnfeaturesX)
    '''
    testImage = "../dataSet/jaffe/Angry/KA.AN3.41.tiff"
    textOnImage(testImage)