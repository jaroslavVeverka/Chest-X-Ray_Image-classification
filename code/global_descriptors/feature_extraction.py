import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mahotas
import h5py

bins = 8

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick


# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

def prepare_images(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([class_num, resized_arr])
            except Exception as e:
                print(e)
        print(f'[STATUS] images from', path, 'prepared')
        print(f'[STATUS] ', class_num)
    return data


def extract_global_features(labeled_images):
    labeled_featured_images = []
    for labeled_image in labeled_images:
        fv_hu_moments = fd_hu_moments(labeled_image[1])
        fv_haralick = fd_haralick(labeled_image[1])
        fv_histogram = fd_histogram(labeled_image[1])

        labeled_featured_images.append(np.hstack([labeled_image[0], fv_hu_moments, fv_haralick, fv_histogram]))

    return labeled_featured_images


train_dir = 'C:/Users/jarda/IdeaProjects/Chest X-Ray_Image classification/images/chest_xray/train'
test_dir = 'C:/Users/jarda/IdeaProjects/Chest X-Ray_Image classification/images/chest_xray/test'
val_dir = 'C:/Users/jarda/IdeaProjects/Chest X-Ray_Image classification/images/chest_xray/val'

labels = ['NORMAL', 'PNEUMONIA']
img_size = 200

train_images = prepare_images(train_dir)
test_images = prepare_images(test_dir)
val_images = prepare_images(val_dir)

print(f'[STATUS] train data size: ', np.array(train_images).shape)
print(f'[STATUS] test data size: ', np.array(test_images).shape)
print(f'[STATUS] validation data size: ', np.array(val_images).shape)

train_extracted_images = extract_global_features(train_images)
test_extracted_images = extract_global_features(test_images)
val_extracted_images = extract_global_features(val_images)

print(f'[STATUS] feature vector size ', np.array(train_extracted_images).shape)
print("[STATUS] feature vector size {}".format(np.array(test_extracted_images).shape))
print("[STATUS] feature vector size {}".format(np.array(val_extracted_images).shape))

h5_train_data = 'C:/Users/jarda/IdeaProjects/Chest X-Ray_Image classification/data/h5_train_data'
h5_test_data = 'C:/Users/jarda/IdeaProjects/Chest X-Ray_Image classification/data/h5_test_data'
h5_val_data = 'C:/Users/jarda/IdeaProjects/Chest X-Ray_Image classification/data/h5_val_data'

h5f_train_data = h5py.File(h5_train_data, 'w')
h5f_train_data.create_dataset('dataset_1', data=pd.DataFrame(train_extracted_images))

h5f_test_data = h5py.File(h5_test_data, 'w')
h5f_test_data.create_dataset('dataset_1', data=pd.DataFrame(test_extracted_images))

h5f_val_data = h5py.File(h5_val_data, 'w')
h5f_val_data.create_dataset('dataset_1', data=pd.DataFrame(val_extracted_images))

h5f_train_data.close()
h5f_test_data.close()
h5f_val_data.close()