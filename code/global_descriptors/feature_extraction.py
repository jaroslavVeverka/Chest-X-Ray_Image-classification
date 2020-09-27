import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_dir = 'C:/Users/jarda/IdeaProjects/Chest X-Ray_Image classification/images/chest_xray/train'
test_dir = 'C:/Users/jarda/IdeaProjects/Chest X-Ray_Image classification/images/chest_xray/test'
val_dir = 'C:/Users/jarda/IdeaProjects/Chest X-Ray_Image classification/images/chest_xray/val'

labels = ['PNEUMONIA', 'NORMAL']
img_size = 200


def prepare_images(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        print(isinstance(class_num, int))
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([class_num, resized_arr])
            except Exception as e:
                print(e)
    return data


train_images = prepare_images(test_dir)
#test_images = prepare_images(test_dir)
#val_images = prepare_images(val_dir)


print("[STATUS] train data size: {}".format(np.array(train_images).shape))
#print("[STATUS] test data size: {}".format(np.array(test_images).shape))
#print("[STATUS] validation data size: {}".format(np.array(val_images).shape))

l = []
num_0 = 0
num_1 = 1
for i in train_images:
    if i[0] == 0:
        l.append("Pneumonia")
        num_0 = num_0 + 1
    else:
        l.append("Normal")
        num_1 = num_1 + 1

plt.hist(l, bins=2)
plt.show()

plt.figure(figsize = (5,5))
plt.imshow(np.array(train_images[0][1]), cmap='gray')
plt.title(labels[np.array(train_images[0][0])])
plt.show()

plt.figure(figsize = (5,5))
plt.imshow(np.array(train_images[-1][1]), cmap='gray')
plt.title(labels[np.array(train_images[-1][0])])
plt.show()

