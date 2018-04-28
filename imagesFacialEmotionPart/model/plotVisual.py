#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 22:19:32 2018

@author: fubao
"""

from keras.utils import plot_model

#from LSTMTrain import lstmModel
from tlFeatureExtract import modifiedVGG16Model

from keras.callbacks import Callback
from matplotlib import pyplot as plt


# https://medium.com/datalogue/attention-in-keras-1892773a4f22
# 

def plotModel():
    '''
    plot the pretrained model and lstm model
    '''
    
    cnnmodel = modifiedVGG16Model(weights_path=None, shape=(1, 48, 48))
    plot_model(cnnmodel, to_file='plots/LSTM_model.svg')

    lstmmodel = lstmModel(trainX_shape=(10000,1, 512), trainY_shape=(10000, 7))
    plot_model(lstmmodel, show_shapes=True, to_file='plots/cnnmodel.svg')



def plotLossAccur(history):
    # plot train and validation loss
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig("plots/LossEpoch")
    plt.show()
    
    plt.figure(2)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model train vs validation acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig("plots/AccuracyEpoch")

    plt.show()
   

if __name__ == "__main__":
    plotModel()
