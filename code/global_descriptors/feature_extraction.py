import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_dir = 'C:/Users/jarda/IdeaProjects/Chest X-Ray_Image classification/images/chest_xray/train'
test_dir = 'C:/Users/jarda/IdeaProjects/Chest X-Ray_Image classification/images/chest_xray/test'
val_dir = 'C:/Users/jarda/IdeaProjects/Chest X-Ray_Image classification/images/chest_xray/val'

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150


def prepare_images(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        print(isinstance(class_num, int))
        for img in os.listdir(path):
            try:
                #img_arr = cv2.imread(os.path.join(path, img))
                #resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([img, class_num])
            except Exception as e:
                print(e)
    return data


train_data = prepare_images(train_dir)


print("[STATUS] train data size: {}".format(np.array(train_data).shape))

l = []
num_0 = 0
num_1 = 1
for i in train_data:
    if i[1] == 0:
        l.append("Pneumonia")
        num_0 = num_0 + 1
    else:
        l.append("Normal")
        num_1 = num_1 + 1

plt.hist(l, bins=2)
plt.show()


