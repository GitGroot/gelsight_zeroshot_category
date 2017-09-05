import os
import cv2
import numpy as np
from config.config import test_large_label, train_large_label, data_path
class Data():
    def __init__(self,attr_num, path=data_path, target_size=224, mode='gray'):
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
        S = self.get_train_attr_label().T if train else self.get_test_attr_label().T
        return S/[(S[:,i]**2).sum()**0.5 for i in range(S.shape[1])]

    def get_large_label(self, little_label):
        for i in range(8):
            if little_label in self.large_classify[i]:
                return i
        return -1

    def load_gelsight_data(self):
        video_list = os.listdir(self.gelsight_path)
        #video_list.sort()
        videos = []
        videos_labels = []
        random_index = range(len(video_list))
        for index in random_index:
            video_folder = video_list[index]
            name = video_folder.split('.')[0]
            class_info = name.split('_')[0].split('F')[1]  # F0001~F0119
            label = eval(class_info.lstrip('0'))
            frame_list = os.listdir(os.path.join(self.gelsight_path, video_folder))
            frame_list.sort()
            frames = []
            for frame_name in frame_list:
                image = cv2.imread(os.path.join(self.gelsight_path, video_folder, frame_name))
                if self.target_size != 224:
                    image = cv2.resize(image, (self.target_size, self.target_size))
                #image = image.astype('float32') / 255.0
                frames.append(image)
            videos.append(frames)
            videos_labels.append(label)
        return videos, videos_labels

    def get_train_test(self):
        videos, videos_labels = self.load_gelsight_data()
        train_videos = []
        train_videos_attr_labels = []
        train_videos_labels = []
        test_videos = []
        test_videos_labels = []
        for i in range( len(videos_labels) ):
            little_label = videos_labels[i]
            large_label = self.get_large_label(little_label)
            if large_label in self.train_large_labels:
                train_videos.append(videos[i])
                train_videos_labels.append(self.train_large_labels.index(large_label))
                train_videos_attr_labels.append(self.attribute_labels[large_label])
            elif large_label in self.test_large_labels:
                test_videos.append(videos[i])
                test_videos_labels.append(self.test_large_labels.index(large_label))

        train_videos = np.array(train_videos).reshape(len(train_videos), self.num_step, \
                                                      self.target_size, self.target_size,self.image_channel)
        test_videos = np.array(test_videos).reshape(len(test_videos), self.num_step, \
                                                      self.target_size, self.target_size,self.image_channel)
        return train_videos, train_videos_labels, train_videos_attr_labels, test_videos, test_videos_labels
