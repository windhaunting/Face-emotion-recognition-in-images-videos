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
from matplotlib.ticker import FuncFormatter

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
    plt.savefig("../plots/LossEpoch")
    plt.show()
    
    plt.figure(2)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model train vs validation acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig("../plots/AccuracyEpoch")

    plt.show()
   

def plotDatasetDistribution(perClassDataDic, dataName):
    '''
    plot dataset distribution for each category  
    '''
    
    # plot
    #xLst = [i for i in len(perClassDataDic)]
    xstrLst = []
    yLst = []
    for emotion, val in perClassDataDic.items():
        xstrLst.append(emotion)
        yLst.append(val)
        
    xLst = [i for i in range(len(xstrLst))]
    xlabel = "Emotion"
    ylabel = "Data size"
    xlim = [0, 8]
    ylim = [0, 1500]
    print ("plotDatasetDistribution perClassDataDic ", perClassDataDic)
    title = ""  # " Runtime vs Query graph size"
    
    plt.figure(1)
    #ax.yaxis.set_major_formatter(formatter)
    
    plt.bar(xLst, yLst, color = 'b')
    plt.xticks(xLst, xstrLst, fontsize = 11)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    
    saveFilePath = '../plots/' + dataName + '-distribution.pdf'    
    plt.savefig(saveFilePath)
    plt.show()
    
if __name__ == "__main__":
    x = 1
    #plotModel()
