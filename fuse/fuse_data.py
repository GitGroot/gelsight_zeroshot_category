from config.fuse_config.config import test_large_label, train_large_label, data_path
import numpy as np
import os
from random import choice
import cv2
import random

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
    def __init__(self,attr_num):
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
        self.num_step = 5
        self.target_size = 224
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

    def get_file_name_list(self):
        little_label_video_table = []
        little_label_depima_table = []
        for i in range(119):
            little_label_video_table.append([])
            little_label_depima_table.append([])
        video_name_list = os.listdir(data_path)
        depima_name_list = os.listdir('../data/depth224')
        for video_name in video_name_list:
            name = video_name.split('.')[0]
            class_info = name.split('_')[0].split('F')[1]  # F0001~F0119
            little_label = eval(class_info.lstrip('0'))
            little_label_video_table[little_label-1].append(os.path.join(data_path, video_name))
        for depima_name in depima_name_list:
            name = depima_name.split('.')[0]
            class_info = name.split('_')[0].split('F')[1]  # F0001~F0119
            little_label = eval(class_info.lstrip('0'))
            little_label_depima_table[little_label-1].append(os.path.join('../data/depth224/', depima_name))
        return little_label_video_table, little_label_depima_table

    def combine(self, a, b):
        ra = []
        rb = []
        for ele1 in a:
            for ele2 in b:
                ra.append(ele1)
                rb.append(ele2)
        return ra, rb

    def randomchoose(self, a, b):
        ra = []
        rb = []
        for ele in a:
            ra.append(ele)
            rb.append(choice(b))
        return ra, rb

    def shuffle_using_same_role1(self, a, b, c):
        i = len(a) - 1
        while i > 0:
            pos = random.randint(0,i-1)
            a[i], a[pos] = a[pos], a[i]
            b[i], b[pos] = b[pos], b[i]
            c[i], c[pos] = c[pos], c[i]
            i -= 1

    def shuffle_using_same_role2(self, a, b, c, d):
        i = len(a) - 1
        while i > 0:
            pos = random.randint(0,i-1)
            a[i], a[pos] = a[pos], a[i]
            b[i], b[pos] = b[pos], b[i]
            c[i], c[pos] = c[pos], c[i]
            d[i], d[pos] = d[pos], d[i]
            i -= 1
# ---------------------------------------------------------------------------------------------
    def get_train_test(self):
        little_label_video_table, little_label_depima_table = self.get_file_name_list()
        train_videos = []
        train_depimas = []
        train_attr_labels = []
        train_labels = []

        test_videos = []
        test_depimas = []
        test_labels = []
        for little_label in range(119):
            large_label = self.get_large_label(little_label+1)
            if large_label in self.train_large_labels:
                tempa, tempb = self.combine(little_label_video_table[little_label], \
                                            little_label_depima_table[little_label])
                train_videos.extend(tempa)
                train_depimas.extend(tempb)
                train_labels.extend([self.train_large_labels.index(large_label)]*len(tempa))
                train_attr_labels.extend([self.attribute_labels[large_label]]*len(tempa))

            elif large_label in self.test_large_labels:
                tempa, tempb = self.randomchoose(little_label_video_table[little_label], \
                                                 little_label_depima_table[little_label])
                test_videos.extend(tempa)
                test_depimas.extend(tempb)
                test_labels.extend([self.test_large_labels.index(large_label)]*len(tempa))

        self.shuffle_using_same_role2(train_videos, train_depimas, train_labels, train_attr_labels)
        self.shuffle_using_same_role1(test_videos, test_depimas, test_labels)
        return train_videos, train_depimas, train_labels, train_attr_labels, \
               test_videos, test_depimas, test_labels

    def get_content(self, videos_name, depimas_name):
        videos = []
        depimas = []
        for vn in videos_name:
            frame_list = os.listdir(vn)
            frame_list.sort()
            frames = []
            for frame_name in frame_list:
                image = cv2.imread(os.path.join(vn, frame_name))
                frames.append(image)
            videos.append(frames)
        for dn in depimas_name:
            image = cv2.imread(dn, 2)
            image = depth_image_normal(image)
            depimas.append(image)
        videos = np.array(videos).reshape(len(videos), self.num_step, self.target_size, self.target_size, 3)
        depimas = np.array(depimas).reshape(len(depimas), self.target_size, self.target_size, 1)
        return videos,  depimas
