import os
import cv2
import numpy as np
from config.config import test_large_label, train_large_label, data_path
from PIL import Image

def depth_image_normal(depth_image):
    a = list(depth_image.reshape(-1))
    a.sort()
    for e in a:
        if e != 0:
            break
    ret = (depth_image-float(e))/float(depth_image.max()-float(e))*255.0
    ret[ret<0]=0
    ret = ret - ret.mean()
    return ret

class Data():
    def __init__(self,attr_num, path='../data/depth224', target_size=224, mode='gray'):
        self.gelsight_path = path
        self.large_classify = [
            [36, 93, 2, 64, 14, 72, 73, 78],
            [63, 59, 27, 98, 97, 13, 56, 69, 113, 24, 21, 53, 108, 12, 115, 118, 5, 94],
            [112, 20, 31, 32, 50, 30, 16, 28, 15, 10, 19, 110, 22, 104, 107, 117, 114, 116, 4, 9, 3, 109, 79, 75, 1],
            [17, 29, 33, 106, 6, 77, 7, 43, 40, 61, 26, 34, 48, 44, 46, 102, 76, 80, 89, 99],
            [91, 62, 88, 55, 51, 47, 8, 87, 66],
            [92, 95, 70, 65, 85, 68],
            [90, 96, 52, 67, 86, 105, 45, 18, 60, 25, 41, 39, 37, 103, 111, 23, 42, 81, 49, 38],
            [82, 57, 58, 100, 74, 54, 71, 83, 84, 101, 35, 11, 119]
        ]
        self.test_large_labels = test_large_label
        self.train_large_labels = train_large_label
        self.target_size = target_size
        self.mode = mode
        self.num_step = 5
        self.image_channel = 1 if mode=='gray' else 3
        if attr_num == 8:
            self.attribute_labels = np.load('../data/a8.npy')
        elif attr_num == 12:
            self.attribute_labels = np.load('../data/a12.npy')
        else:
            print('error')
            exit()

    def get_train_attr_label(self):
        ret = []
        for large_label in self.train_large_labels:
            ret.append(self.attribute_labels[large_label])
        return np.array(ret)

    def get_test_attr_label(self):
        ret = []
        for large_label in self.test_large_labels:
            ret.append(self.attribute_labels[large_label])
        return np.array(ret)

    def S(self,train):
        return self.get_train_attr_label().T if train else self.get_test_attr_label().T

    def get_large_label(self, little_label):
        for i in range(8):
            if little_label in self.large_classify[i]:
                return i
        return -1

    def load_gelsight_data(self):
        depima_list = os.listdir(self.gelsight_path)
        #video_list.sort()
        depimas = []
        depimas_labels = []
        random_index = range(len(depima_list))
        for index in random_index:
            depima_name = depima_list[index]
            name = depima_name.split('.')[0]
            class_info = name.split('_')[0].split('F')[1]  # F0001~F0119
            label = eval(class_info.lstrip('0'))
            image = cv2.imread(os.path.join(self.gelsight_path, depima_name), 2)
            image = depth_image_normal(image)
            depimas.append(image)
            depimas_labels.append(label)
        return depimas, depimas_labels

    def get_train_test(self):
        depimas, depimas_labels = self.load_gelsight_data()
        train_depimas = []
        train_depimas_attr_labels = []
        train_depimas_labels = []
        test_depimas = []
        test_depimas_labels = []
        for i in range( len(depimas_labels) ):
            little_label = depimas_labels[i]
            large_label = self.get_large_label(little_label)
            if large_label in self.train_large_labels:
                train_depimas.append(depimas[i])
                train_depimas_labels.append(self.train_large_labels.index(large_label))
                train_depimas_attr_labels.append(self.attribute_labels[large_label])
            elif large_label in self.test_large_labels:
                test_depimas.append(depimas[i])
                test_depimas_labels.append(self.test_large_labels.index(large_label))

        train_depimas = np.array(train_depimas).reshape(len(train_depimas), \
                                                      self.target_size, self.target_size,self.image_channel)
        test_depimas = np.array(test_depimas).reshape(len(test_depimas), \
                                                      self.target_size, self.target_size,self.image_channel)
        return train_depimas, train_depimas_labels, train_depimas_attr_labels, test_depimas, test_depimas_labels

ld = Data(8)
ld.load_gelsight_data()