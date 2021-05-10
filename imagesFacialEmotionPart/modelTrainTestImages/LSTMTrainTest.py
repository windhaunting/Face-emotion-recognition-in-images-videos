#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 19:56:00 2018

@author: fubao
"""

'''
LSTM model to train facial emotions dataset

'''

           
# 1 dataset : JAFFE
import numpy as np
from sklearn.model_selection import StratifiedKFold


from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout,Flatten
from keras.layers import LSTM
from plotVisual import plotLossAccur
from keras import backend as Kb
from keras import optimizers

from tlFeatureExtract import extractModifiedVGGArray

from dataPreprocess import getJAFFEDataArrayFromImages
from dataPreprocess import getCKDataArrayFromImages

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from plotVisual import plotconfusionMatrix 

def lstmModel(trainX_shape, trainY_shape, weights_path=None):
    
    
    print ("lstmModel trainx shape ", trainX_shape, trainY_shape)
    max_features = 1000
    embed_outDim = 64
    input_length = trainX_shape[0]  # 213
    
    time_step = trainX_shape[1] #1 #trainX_shape[1]      # 48
    features = trainX_shape[2]       #48   # or time_step 1, features = 48*48= 2304
    model = Sequential()
    #model.add(Embedding(max_features, output_dim = embed_outDim, input_length = input_length))
    model.add(LSTM(256, input_shape=(time_step, features), return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(128, input_shape=(time_step, 128), return_sequences=True))
    model.add(Dropout(0.5))
    
    weights = model.layers[0].get_weights()
    print(model.summary())
    print ("first layer weight: ",len(weights), np.asarray(weights).shape)
    
    model.add(Flatten())
    print(model.summary())

    #model.add(Dense(213))
    #model.add(Dropout(0.2))
    
    #model.add(LSTM(128))
    #model.add(Dropout(0.2))
    #model.add(Flatten())

    model.add(Dense(trainY_shape[1], activation='softmax'))

    # try using different optimizers and different optimizer configs
    adam = optimizers.Adam(lr=0.0001, decay=1e-6)
    #Kb.set_value(model.optimizer.lr, 0.01)
    
    if weights_path:
        model.load_weights(weights_path)
        
        
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,   #'adam',
                  metrics=['accuracy'])
    
    print ("lstmModel created successfully learning rate: ", Kb.get_value(model.optimizer.lr))
    return model

def lstmTrain(feature_x, y):
    
    '''
    train on lstm used trainsferred learning from cnn feature extractor
    '''
    x_train = feature_x # [:50]   #  feature_x
    #x_train = x_train.reshape((x_train.shape[0], -1))
    y_train = y#  [:50]    # y_train
    print(x_train.shape)
    print(y_train.shape) 
    batch_size = 10
    
    trainX_shape, trainY_shape = x_train.shape, y_train.shape
    model = lstmModel(trainX_shape, trainY_shape)

    # transfer learning from cnn
    '''
    # define 10-fold cross validation test harness
    # https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(x_train, y_train):
    
    '''
    print('Training...')
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=100,
              validation_split=0.2,
              shuffle=True,
              #callbacks=[PlotLearning],
              verbose=2)
    print ("------*****")

    print(history.history['val_loss'])

    #plot loss 
    plotLossAccur(history)
    
    model.save_weights('lstm_model_weights.h5')
    scores = model.evaluate(x_train, y_train, verbose=0)
    print ("Train loss : %.3f" % scores[0])
    print ("Train accuracy : %.3f" % scores[1])
    print ("Training finished")
    
    return model
    
def lstmTrainExecute(datasetId):
    #train JAFFE dataset
    if datasetId == 0:
        x = np.load('dataNumpy/JAFFE_trainvalidation_X.npy')
        y = np.load('dataNumpy/JAFFE_trainvalidation_y.npy')
        
    elif datasetId == 1:
        #train CK+ dataset
        x = np.load('dataNumpy/CK+_trainvalidation_X.npy')
        y = np.load('dataNumpy/CK+_trainvalidation_y.npy')
        
    else: 
        # combined dateset
        x = np.load('dataNumpy/total_trainvalidation_X.npy')
        y = np.load('dataNumpy/total_trainvalidation_y.npy')
        

    # split for train and validation, split for test

    #zero mean and standardlization 
    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)
    
    print('getJAFFEDataArrayFromImages x.shape, y.shape: ', x.shape, y.shape)
    # w/o pretrained CNN model
    #x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
    
    #x = np.stack((x,)*3, 1)

    #use original images
    #x = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    #print('get_dataArrayFromImages x.shape, y.shape: ', x.shape, y.shape)
 
    #by cnn feature extractor transferred learning
    x = extractModifiedVGGArray(x, shape=x.shape[1:])   #(x, shape = x.shape[1:])
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
    print('lstmTrainExecute use pretrained x.shape, y.shape: ', x.shape, y.shape)


    model = lstmTrain(x, y)
    
    return model
    
def testModel(datasetId, modelWeight):      
    
    #cm = confusion_matrix(y_test, y_pred)

    if datasetId == 0:
        x_test = np.load('dataNumpy/JAFFE_test_X.npy')
        y_test = np.load('dataNumpy/JAFFE_test_y.npy')
        x_test -= np.mean(x_test, axis=0)
        x_test /= np.std(x_test, axis=0)
    
        # 
        print('testModelJAFFE x.shape, y.shape: ', x_test.shape, y_test.shape)
        
        #use pretrained feature too
        x_test = extractModifiedVGGArray(x_test, shape=x_test.shape[1:])   #(x, shape = x.shape[1:])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2]*x_test.shape[3])
    
        dataName = "JAFFE"

    elif datasetId == 1:
        #train CK+ dataset
        #feature_x, y = getCKDataArrayFromImages()
        x_test = np.load('dataNumpy/CK+_test_X.npy')
        y_test = np.load('dataNumpy/CK+_test_y.npy')
        
        print('testModel CK+ x.shape, y.shape: ', x_test.shape, y_test.shape)
        x_test = extractModifiedVGGArray(x_test, shape=x_test.shape[1:])   #(x, shape = x.shape[1:])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2]*x_test.shape[3])
  
        dataName = "CK"
    elif datasetId == 2:
        x_test = np.load('dataNumpy/total_test_X.npy')
        y_test = np.load('dataNumpy/total_test_y.npy')
        
        #without cnn feature
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2], x_test.shape[3])


        print('testModel total x.shape, y.shape: ', x_test.shape, y_test.shape)
        #x_test = extractModifiedVGGArray(x_test, shape=x_test.shape[1:])   #(x, shape = x.shape[1:])
        #x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2]*x_test.shape[3])
  
        dataName = "totalDataset"
        
    model = lstmModel(x_test.shape, y_test.shape, modelWeight)
    y_pred = model.predict(x_test)

    y_test = np.argmax(y_test, axis=1) # Convert one-hot to index
    y_pred = model.predict_classes(x_test)
    
    #print ("y_test: ", list(y_test))
    #print ("y_pred: ", list(y_pred))
    print(classification_report(y_test, y_pred))
    
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
   
    plotconfusionMatrix(y_test, y_pred, labels, dataName)

if __name__ == "__main__":
    #testImage = "../dataSet/jaffe/KA.AN3.41.tiff"
    #dataDirPathJAFFE = "../dataSet/jaffe/"
    #lstmTrainExecute( 0)
    
    # test
    #modelWeights = 'JAFFE_lstm_model_weights.h5'
    #testModel(0, modelWeights)
    
    #modelWeights = 'JAFFE_pretrain_lstm_model_weights.h5'
    #testModel(0, modelWeights)

    # CK+ 
    #dataPathCK = "../dataSet/CK+/cohn-kanade-images/"
    #lstmTrainExecute(1)

    # test
    #modelWeights = 'CK+_lstm_model_weights.h5'
    #testModel(1, modelWeights)
    
    #lstmTrainExecute(2)
    modelWeights = "total_with_lstm_model_weights.h5"
    testModel(2, modelWeights)
    