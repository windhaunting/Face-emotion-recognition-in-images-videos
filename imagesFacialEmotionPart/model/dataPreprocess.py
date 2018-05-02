#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 09:54:14 2018

@author: fubao
"""

import sys, os
import cv2
import numpy as np

# load data of images to transfer to numpy array used for training/validation/testing

# JAFFE  data

from collections import defaultdict	

from keras.preprocessing.image import ImageDataGenerator

from tlFeatureExtract import preprocessing
from tlFeatureExtract import extractModifiedVGGSingleImages 
from tlFeatureExtract import extractModifiedVGGArray
from tlFeatureExtract import extractModifiedVGG16toArray
from tlFeatureExtract import visualLayersOutput

from CNNActivation import display_activations


from plotVisual import plotDatasetDistribution

emotions = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
           'Sad': 4, 'Surprise': 5, 'Neutral': 6}


def shuffle_arrays_unison(arrays, random_state=None):
    '''
    Shuffle NumPy arrays in unison.
    Parameters
    ----------
    arrays : array-like, shape = [n_arrays]
      A list of NumPy arrays.
    random_state : int
      Sets the random state.
    Returns
    '''
    
    if random_state:
        np.random.seed(random_state)
    n = len(arrays[0])
    for a in arrays:
        assert(len(a) == n)
    idx = np.random.permutation(n)
    return [a[idx] for a in arrays]
            
def getJAFFEDataArrayFromImages(dataDirPath):
    '''
    get x input from images by cv2 preprocessing and cnn pretrained model
    '''
    x, y = [], []
    #label = 0
    perClassDataDic = defaultdict(int)       # number of each class labels;  key: emotion, values: count
    
    cnt = 0
    for dirnames in os.listdir(dataDirPath):
        # print(dirnames)
        sub_path = os.path.join(dataDirPath, dirnames)
        #print(sub_path)

        for filename in os.listdir(sub_path):
            file_path = os.path.join(sub_path, filename)
            #print (file_path)
            #img = cv2.imread(file_path)
            img = preprocessing(file_path, size=(48,48))
            
            #cnt += 1
            #if cnt > 100:
            #    break
            
            # transferred learning feature
            x.append(img)
            
            #get label_y
            emo_file = sub_path.split("/")[-1]
            classes = [0, 0, 0, 0, 0, 0, 0]
            classes[emotions[emo_file]] = 1
            y.append(classes)
             
            #print ('xxxxxxxx: ', filename, x, y)
            perClassDataDic[emo_file] += 1

    
    
    x = np.asarray(x)
    y = np.asarray(y)
    print ("JAFFE file numy cnt: ", cnt, x.shape, y.shape)               # total shape (5643, 7

    
    '''
    x,y = shuffle_arrays_unison([x, y])
    
    x_train = x[:int(0.8*len(x))]
    y_train = y[:int(0.8*len(y))]
    
    np.save('dataNumpy/JAFFE_trainvalidation_X.npy', x_train)
    np.save('dataNumpy/JAFFE_trainvalidation_y.npy', y_train)
    
        
    x_test = x[int(0.8*len(x)):]
    y_test = y[int(0.8*len(y)):]
    
    np.save('dataNumpy/JAFFE_test_X.npy', x_test)
    np.save('dataNumpy/JAFFE_test_y.npy', y_test)
    
    '''
    
    '''
    print ("perclassDataDic: ", perClassDataDic)
    #plotDatasetDistribution(perClassDataDic, "JAFFE")
    '''
    
    return x, y



# second data CK+
def getCKDataArrayFromImages(dataDirPath):
    '''
    # CK+ emotions 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise)
    '''
    
    inputEmotions = {0: 'Neutral', 1: 'Angry', 2: 'Contempt', 3: 'Disgust',
                     4: 'Fear', 5: 'Happy', 6: 'Sad', 7: 'Surprise'}
    # we only use 7 emotions
    #read images and also transfer to our emotions categories
    
    x, y = [], []

    cnt = 0
    
    perClassDataDic = defaultdict(int)       # number of each class labels;  key: emotion, values: count
    
    for dirnames1 in os.listdir(dataDirPath):
        # print(dirnames)
        sub_path1 = os.path.join(dataDirPath, dirnames1)
        #print(sub_path1)
        for dirnames2 in os.listdir(sub_path1):
            sub_path2 = os.path.join(sub_path1, dirnames2)
            #print('sub_path2: ', sub_path2)

            #for filename in os.listdir(sub_path2):
            #file_path = os.path.join(sub_path2, filename)
            #print (file_path)
            
            emotionLabelFilePath = '/'.join(sub_path1.split('/')[:3]) + '/Emotion/' + dirnames1 + '/' + dirnames2
            if os.path.isdir(emotionLabelFilePath):
                filenameLst = sorted(list(os.listdir(emotionLabelFilePath)))
                #print ("filenameLst: ", filenameLst)
            
                for filename in os.listdir(emotionLabelFilePath):
                    emotionLabelPathFile = emotionLabelFilePath + '/' + filename
                    #print ("emotionLabelPathFile: ", emotionLabelPathFile)
                    
                    cnt += 1
                    
                    #if cnt > 3:            #total 5876
                    #    break
                    
                    #1st get the last N figure as the emotion_labeled feature
                    emotionLabelFile = filename
                    #print ("emotionLabelFile: ", emotionLabelFile)
                    emoLabel_figNo = emotionLabelFile.split('_')[2]
                    
                    #use last four emotion
                    emotionLabel =  int(float(open(emotionLabelPathFile).read().strip().lower()))  # read contents from file
                    # transfer different emotions representation
                    if emotionLabel == 2:     # do not consider "contempt
                        continue
                    emotionId = emotions[inputEmotions[emotionLabel]]
                    
                    figure_path = os.path.join(sub_path2)
                    figuresNames = sorted([f for f in os.listdir(figure_path) if f.endswith('.png')])
                    N = int(len(figuresNames)/3)           # 1/3 
                    
                    for figName in figuresNames[::-1][:N]:
                        classes = [0, 0, 0, 0, 0, 0, 0]
                        #print ("emotionLabel: ", emotionLabel, emotionId)
                        classes[emotionId] = 1
                        y.append(classes)
                        #print ("figures: ", figName)
                        #emotional labeled file 
                        #figFileName = '_'.join(emotionLabelFile.split('_')[:2]) + '_' + emoLabel_figNo + '.png'
                        imgFile = os.path.join(sub_path2, figName)
                        #print ("imgFile: ", imgFile)
                        img = preprocessing(imgFile, size=(48, 48))
                        x.append(img)
                        
                        perClassDataDic[inputEmotions[emotionLabel]] += 1
                        
                    #2nd get neutral, use first 2nd figures as neutral
                    N = 2
                    for figName in figuresNames[:N]:
                        imgFileNeutral = os.path.join(sub_path2, figName)
                        imgNeu = preprocessing(imgFileNeutral, size=(48, 48))
                        #print ("imgFile22: ", imgFileNeutral)
                        x.append(imgNeu)
                        classes = [0, 0, 0, 0, 0, 0, 0]
                        emotionId = 6
                        classes[emotionId] = 1
                        y.append(classes)

                        perClassDataDic[inputEmotions[0]] += 1
                        
    #print ("CK+file aaaa numy cnt: ", cnt, len(x), len(x))               # total shape (5643, 7
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    print ("CK+ file numy cnt: ", cnt, x.shape, y.shape)               # total shape (5643, 7

    '''
    x,y = shuffle_arrays_unison((x, y))      # randomly shuffle dataset
    
    x_train = x[:int(0.8*len(x))]
    y_train = y[:int(0.8*len(y))]
    
    np.save('dataNumpy/CK+_trainvalidation_X.npy', x_train)
    np.save('dataNumpy/CK+_trainvalidation_y.npy', y_train)
    
        
    x_test = x[int(0.8*len(x)):]
    y_test = y[int(0.8*len(y)):]
    
    np.save('dataNumpy/CK+_test_X.npy', x_test)
    np.save('dataNumpy/CK+_test_y.npy', y_test)

    print ("perclassDataDic: ", perClassDataDic)
    '''
    
    #plotDatasetDistribution(perClassDataDic, "CK+")
    
    return x, y
    

def getAllDataTrainResult(dataDirPathJA, dataDirPathCK):
    '''
    combine the two dataset JAFFE + CK+
    '''
    xJA, yJA = getJAFFEDataArrayFromImages(dataDirPathJA)
    xCK, yCK = getCKDataArrayFromImages(dataDirPathCK)
    
    x= np.vstack((xJA, xCK))
    y= np.vstack((yJA, yCK))
    
    
    x,y = shuffle_arrays_unison((x, y))      # randomly shuffle dataset
    
    x_train = x[:int(0.8*len(x))]
    y_train = y[:int(0.8*len(y))]
    
    np.save('dataNumpy/total_trainvalidation_X.npy', x_train)
    np.save('dataNumpy/total_trainvalidation_y.npy', y_train)
    
        
    x_test = x[int(0.8*len(x)):]
    y_test = y[int(0.8*len(y)):]
    
    np.save('dataNumpy/total_test_X.npy', x_test)
    np.save('dataNumpy/total_test_y.npy', y_test)

    print ("getAllDataTrainResultfile numy: ", x.shape, y.shape)    
    
    
    
    
    
if __name__ == "__main__" :
    #dataPathJAFFE =  "../dataSet/jaffe/"
    #x, y = getJAFFEDataArrayFromImages(dataPathJAFFE)
    #print(x.shape, y.shape)
    #print(type(x), type(x[0]))
    #print(x[212])
    #print(len(X))
    
    #dataPathCK = "../dataSet/CK+/cohn-kanade-images/"
    #getCKDataArrayFromImages(dataPathCK)
    
    
    # show one image
    '''
    testImage = "../dataSet/jaffe/Angry/KA.AN3.41.tiff"

    img = preprocessing(testImage)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    
    '''
    #testImage = "../dataSet/jaffe/Angry/KA.AN3.41.tiff"
 
    #visualLayersOutput(testImage)
    #testImage =  "../dataSet/jaffe/Angry/KA.AN3.41.tiff"
    #x = extractModifiedVGGSingleImages(testImage, shape=(1, 256, 256))
    #print ("xxxxxxxx shape: ", x.shape)
    #display_activations(x)

    dataPathJAFFE =  "../dataSet/jaffe/"
    dataPathCK = "../dataSet/CK+/cohn-kanade-images/"
    getAllDataTrainResult(dataPathJAFFE, dataPathCK)
    