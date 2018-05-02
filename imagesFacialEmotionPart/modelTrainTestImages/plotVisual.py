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
from sklearn.metrics import confusion_matrix


# https://medium.com/datalogue/attention-in-keras-1892773a4f22
# 

def plotModel():
    '''
    plot the pretrained model and lstm model
    '''
    
    cnnmodel = modifiedVGG16Model(weights_path=None, shape=(1, 256, 256))
    plot_model(cnnmodel, show_shapes=True, to_file='../plots/CNN_model.svg')

    #lstmmodel = lstmModel(trainX_shape=(10000,1, 512), trainY_shape=(10000, 7))
    #plot_model(lstmmodel, show_shapes=True, to_file='../plots/LSTM_model.svg')


def plotLossAccur(history):
    # plot train and validation loss
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig("../plots/LossEpoch.pdf")
    plt.show()
    
    plt.figure(2)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model train vs validation acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig("../plots/AccuracyEpoch.pdf")

    plt.show()
   

def plotDatasetDistribution(perClassDataDic, dataName):
    '''
    plot dataset distribution for each category  
    
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    
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
  
def plotconfusionMatrix(y_true, y_pred, labels, dataName):
    #cm = confusion_matrix(y_true, y_pred)
    #labels = ['business', 'health', 'sports']
    cm = confusion_matrix(y_true, y_pred)  # labels)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    #plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels, fontsize=8, style='italic')
    ax.set_yticklabels([''] + labels, fontsize=8)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('True', fontsize=16)
        
    saveFilePath = '../plots/' + dataName + '-confusionMatrix22.pdf'    
    plt.savefig(saveFilePath)
    plt.show()
    



if __name__ == "__main__":
    x = 1
    #plotModel()

    