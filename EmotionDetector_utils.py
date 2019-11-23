import keras.backend as K
import  math
import numpy as np
import h5py
import matplotlib.pyplot as plt

def load_dataset():
    train_dataset=h5py.File("datasets/Emotion_Detector/train_happy.h5",'r')
    test_dataset=h5py.File("datasets/Emotion_Detector/test_happy.h5",'r')
    X_train_orig=np.array(train_dataset["train_set_x"][:])
    Y_train=np.array(train_dataset["train_set_y"][:])
    
    X_test_orig=np.array(test_dataset["test_set_x"][:])
    Y_test=np.array(test_dataset["test_set_y"][:])
    
    Y_train_orig=Y_train.reshape((1,Y_train.shape[0]))
    Y_test_orig=Y_test.reshape((1,Y_test.shape[0]))
    
    classes=np.array(test_dataset["list_classes"][:])
    return X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes