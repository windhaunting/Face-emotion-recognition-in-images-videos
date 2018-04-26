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

from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM


def LstmModel():
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    
    model.add(Dense(1, activation='sigmoid'))
    
    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
def lstmTrain():
    
    '''
    train on lstm used trainsferred learning from cnn feature extractor
    
    '''
    

    
    
    batch_size = 10
    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=15,
              validation_split=0.2,
              shuffle=True,
              verbose=0)
    
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Train score:', score)
    print('Train accuracy:', acc)
