# -*- coding: utf-8 -*-
"""
Created on Sun May  6 19:03:36 2018

@author: Acer
"""

import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name,res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    return mfccs

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):
    features, labels = np.empty((0,40)), np.zeros((2304,8))
    idx=-1
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            idx=idx+1
            mfccs= extract_feature(fn)
            ext_features = np.hstack([mfccs])
            features = np.vstack([features,ext_features])
            labels[idx][int(fn[50:52])-1] = 1
            print(idx)
            print(labels)
            #print(fn[50:52])
        print(labels)
        print(features)
    return np.array(features), np.array(labels, dtype = np.int)


parent_dir = 'C:/Users/Acer/Desktop/emotion_data'
tr_sub_dirs = ["Actor_01","Actor_02","Actor_03","Actor_04","Actor_05","Actor_06","Actor_07","Actor_08","Actor_09","Actor_10","Actor_11","Actor_12","Actor_13","Actor_14","Actor_15","Actor_16","Actor_17","Actor_19","Actor_20","Actor_21","Actor_22","Actor_23","Actor_24"]
tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(tr_features,tr_labels,test_size=0.1,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

num_labels = y_train.shape[1]
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.constraints import max_norm

classifier=Sequential()
classifier.add(Dense(output_dim=256,init='uniform',activation='relu',input_dim=40))
classifier.add(Dropout(0.5))

classifier.add(Dense(output_dim=256,init='uniform',activation='relu'))
classifier.add(Dropout(0.5))


classifier.add(Dense(output_dim=8,init='uniform',activation='softmax'))

sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
classifier.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=256,nb_epoch=400,validation_data=(X_test, y_test))

y_pred=classifier.predict(X_test)


y_test_non_category = [ np.argmax(t) for t in y_test ]
y_predict_non_category = [ np.argmax(t) for t in y_pred ]


from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)