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
test_data = pd.DataFrame(test_dataset)
h5f_test_data.close()

print("[STATUS] test shape: {}".format(test_data.shape))
print(f'Number of test data with label 0', sum(test_data[0] == 0))
print(f'Number of test data with label 1', sum(test_data[0] == 1))

trained_models = joblib.load("trained_models.pkl")

print(trained_models)
print("[STATUS] testing started...")

for name, model in trained_models:
    print("[STATUS] start prediction with model of {}".format(name))
    prediction = model.predict(test_data.iloc[:,1:])
    result = accuracy_score(test_data.iloc[:,0], prediction)
    msg = "%s: %f" % (name, result)
    print(msg)
    print(confusion_matrix(prediction, test_data.iloc[:,0]))

    cm = confusion_matrix(prediction, test_data.iloc[:, 0])
    m = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])

    pyplot.figure(figsize = (10,10))
    sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='',xticklabels = labels,yticklabels = labels)
    pyplot.show()


