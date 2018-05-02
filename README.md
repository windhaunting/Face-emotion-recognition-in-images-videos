

Project: face-emotion-recognition-in-images-videos

We explore human recognition system to identify 7 types of emotions by using FER2013 dataset as a pretrained model features.
Furthermor, we apply this CNN pretrained model features to a Long Short-Term Memory(LSTM) networks and using two
datasets(JAFFE and AFEW) to train it and evaluate it. Finally, we use this
model to test on image datasets and video datasets to compare the performance and get relative high accuracy with and without CNN pretrained model features


# Dependencies  
python3 
opencv  

keras


'''
tflearn


Note:  
install python3 opencv:  conda install -c menpo opencv3=3.1.0

tflearn install for python3:

sudo pip3 install --upgrade $TF_BINARY_URL
pip3 install tensorflow
pip3 install git+https://github.com/tflearn/tflearn.git

--reference: http://tflearn.org/installation/




