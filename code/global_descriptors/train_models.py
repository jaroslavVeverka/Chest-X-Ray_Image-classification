import h5py
import numpy as np
import os
import glob
import cv2
import warnings
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

h5_train_data = 'C:/Users/jarda/IdeaProjects/Chest X-Ray_Image classification/data/h5_train_data'
h5f_train_data = h5py.File(h5_train_data, 'r')
train_dataset = h5f_train_data['dataset_1']
train_data = np.array(train_dataset)
h5f_train_data.close()

print("[STATUS] train shape: {}".format(train_data.shape))

print("[STATUS] training started...")

seed = 9
num_trees = 100

# create all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=seed, max_iter=1000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=seed)))

trained_models = []
for name, model in models:
    print("[STATUS] start training model of {}.".format(name))
    model.fit(train_data[:,1:], train_data[:,0])
    trained_models.append((name, model))
    print("[STATUS] model of {} trained".format(name))


print(trained_models)
joblib.dump(trained_models, "trained_models.pkl", compress=3)