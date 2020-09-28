import h5py
import numpy as np
import os
import glob
import cv2
import warnings

import seaborn as sns
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import joblib
import pandas as pd

labels = ['NORMAL', 'PNEUMONIA']

h5_test_data = 'C:/Users/jarda/IdeaProjects/Chest X-Ray_Image classification/data/h5_test_data'
h5f_test_data = h5py.File(h5_test_data, 'r')
test_dataset = h5f_test_data['dataset_1']
test_data = np.array(test_dataset)
h5f_test_data.close()

l = []
num_0 = 0
num_1 = 1
for i in test_data:
    if i[0] == 0:
        l.append("Normal")
        num_0 = num_0 + 1
    else:
        l.append("Pneumonia")
        num_1 = num_1 + 1

print(num_0)
print(num_1)

print("[STATUS] test shape: {}".format(test_data.shape))

trained_models = joblib.load("trained_models.pkl")

print(trained_models)
print("[STATUS] testing started...")

for name, model in trained_models:
    print("[STATUS] start prediction with model of {}".format(name))
    prediction = model.predict(test_data[:,1:])
    result = accuracy_score(test_data[:,0], prediction)
    msg = "%s: %f" % (name, result)
    print(msg)
    print(confusion_matrix(test_data[:,0], prediction))

    cm = confusion_matrix(test_data[:,0], prediction)
    m = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])

    pyplot.figure(figsize = (10,10))
    sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='',xticklabels = labels,yticklabels = labels)
    pyplot.show()


