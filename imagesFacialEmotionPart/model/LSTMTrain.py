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

from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout,Flatten
from keras.layers import LSTM

from dataPreprocess import get_dataArray

def lstmModel(trainX_shape, trainY_shape):
    
    max_features = 1000
    embed_outDim = 64
    input_length = trainX_shape[0]  # 213
    
    time_step = trainX_shape[1]      # 48
    features = trainX_shape[2]       #48   # or time_step 1, features = 48*48= 2304
    model = Sequential()
    #model.add(Embedding(max_features, output_dim = embed_outDim, input_length = input_length))
    model.add(LSTM(128, input_shape=(time_step, features), return_sequences=True))
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
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

def lstmTrain(feature_x, y):
    
    '''
    train on lstm used trainsferred learning from cnn feature extractor
    '''
    x_train = feature_x   # [:100]   #  feature_x
    #x_train = x_train.reshape((x_train.shape[0], -1))
    y_train = y   #  [:100]    # y_train
    print(x_train.shape)
    print(y_train.shape) 
    batch_size = 10
    
    trainX_shape, trainY_shape = x_train.shape, y_train.shape
    model = lstmModel(trainX_shape, trainY_shape)

    # transfer learning from cnn
    
    print('Training...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=15,
              validation_split=0.2,
              shuffle=True,
              verbose=0)
    print ("---------**")
    score, acc = model.evaluate(x_train, y_train,
                                batch_size=batch_size)
    print('Train score:', score)
    print('Train accuracy:', acc)


def lstmTrainExecute(dataDirPath):
    feature_x, y = get_dataArray(dataDirPath)
    lstmTrain(feature_x, y)
    
    
if __name__ == "__main__":
    #testImage = "../dataSet/jaffe/KA.AN3.41.tiff"
    dataDirPath = "../dataSet/jaffe/"
    
    lstmTrainExecute(dataDirPath)