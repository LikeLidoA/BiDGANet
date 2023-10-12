#!/usr/bin/env python
# coding=utf-8

import os
import numpy as np
import cv2
from torch.utils.data import Dataset as BaseDataset
from datasets.augment import get_validation_augmentation, get_training_augmentation
import time


# classes for data loading and preprocessing
class CamVidDataset(BaseDataset):

    def __init__(
            self,
            txt_dir=None,
            classes=None,
            height=720,
            width=960,
            resize=False,
            augmentation=None,
            preprocessing=True,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]):

        self.width = width
        self.height = height
        self.img_filename_list, self.label_filename_list = self.get_filename_list(txt_dir)
        if classes == 11:
            self.class_values = [i for i in range(11)]
        elif classes == 19:
            self.class_values = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24,
                                 25, 26, 27, 28, 31, 32, 33]
        self.mean = mean
        self.std = std
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.resize = resize

    def __getitem__(self, i):

        # read data
        image = cv2.imread("./datasets/CamVid/"+self.img_filename_list[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread("./datasets/CamVid/"+self.label_filename_list[i], 0)

        if self.resize:
            image = cv2.resize(image, (self.width, self.height))
            mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')
        #
        # # add background if mask is not binary
        # if mask.shape[-1] != 1:
        #     background = 1 - mask.sum(axis=-1, keepdims=True)
        #     mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(self.height, self.width)(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            image = self.input_transform(image)
            # mask = self.label_transform(mask)

        return image, mask

    def __len__(self):
        return len(self.img_filename_list)

    def input_transform(self, image):
        image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        # image = image.transpose((2, 0, 1))
        return image

    def label_transform(self, label):
        return label.transpose(2, 0, 1).astype('long')  # float32

    def get_filename_list(self, txt_dir):
        f = open(txt_dir, 'r')
        img_filename_list = []
        label_filename_list = []
        for line in f:
            try:
                line = line[:-1].split(' ')
                img_filename_list.append(line[0])
                label_filename_list.append(line[1])
            except ValueError:
                print('Check that the path is correct.')

        return img_filename_list, label_filename_list


if __name__ == "__main__":



    train_txt = "../datasets/CamVid/camvid_train.txt"
    height, width = 720, 960
    train_dataset = CamVidDataset(txt_dir=train_txt, height=height, width=width, classes=11,
                                  augmentation=get_training_augmentation(height, width))
    t1 = time.time()
    for idx, data in train_dataset:
        print(idx, data)
    print(time.time() - t1)
