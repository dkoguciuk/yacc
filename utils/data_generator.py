#!/usr/bin/python
# -*- coding: utf-8 -*-

# The MIT License (MIT)
# Copyright (c) 2018 Daniel Koguciuk <daniel.koguciuk@gmail.com>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

'''
@author: Daniel Koguciuk <daniel.koguciuk@gmail.com>
@note: Created on 23.03.2018
'''

import os
import sys
import math
import cv2
import h5py
import random
import cPickle
import exceptions
import numpy as np
import numpy.core.umath_tests as nm

TEST_DIR = 'test'
TRAIN_DIR = 'train'
CLASSES_COUNT = 10
DATASET_IMAGES_DIR = 'cifar-10-batches-py'
DATASET_FEATURES_DIR = 'cifar-10-batches-py-cnn-features'

class CifarImages(object):

    def __init__(self, colorspace='BGR'):
        """
        Colorspace: BGR, GRAY
        """
        
        # train data
        filepaths_train = [os.path.join(DATASET_IMAGES_DIR, filename) for filename in os.listdir(DATASET_IMAGES_DIR) if 'data' in filename]
        self.data_images_train, self.data_labels_train = self._read_filepaths(filepaths_train, colorspace)
        
        # test data
        filepaths_test = [os.path.join(DATASET_IMAGES_DIR, filename) for filename in os.listdir(DATASET_IMAGES_DIR) if 'test' in filename]
        self.data_images_test, self.data_labels_test = self._read_filepaths(filepaths_test, colorspace)

    def get_random_images(self, train=True, instance_number=10):
        """
        Take random images for each class with how_many_instances examples of each class.

        Args:
            train (bool): Should I provide you with train or test examples?
            instance_number (int): How many examples of each class do you want?

        Returns:
            (list of np.ndarrays of size [how_many_instances, 32, 32, 3]): Images.
        """
        if train:
            images = self.data_images_train
            labels = self.data_labels_train
        else:
            images = self.data_images_test
            labels = self.data_labels_test
        
        ret = []
        for class_idx in range(0, CLASSES_COUNT):
            
            class_indices = np.squeeze(np.argwhere(labels == class_idx))
            class_indices_random = np.random.choice(class_indices, size=instance_number)
            class_images = np.take(images, class_indices_random, axis=0)            
            ret.append(class_images)
        
        return ret        

    def generate_batch(self, train, batch_size):
        """
        Generate representative batch of images.

        Args:
            train (bool): Should I provide you with train or test examples?
            batch_szie (int): How many samples do you want?

        Returns:
            (np.ndarray of size [batch_size, 32, 32, 3],
             np.ndarray of size [batch_size, 1]): Images and their labels.
        """
        if train:
            images = self.data_images_train
            labels = self.data_labels_train
        else:
            images = self.data_images_test
            labels = self.data_labels_test

        indices = np.arange(images.shape[0])
        np.random.shuffle(indices)
        
        for batch_idx in range(images.shape[0] / batch_size):
            
            batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_images = images[batch_indices, ...]
            batch_labels = labels[batch_indices]
            yield batch_images, batch_labels

    def size(self, train=True):
        if train:
            images = self.data_images_train
        else:
            images = self.data_images_test
            
        return images.shape[0]

    def _read_filepaths(self, filepaths, colorspace):
        
        # vars 
        data_images = None
        data_labels = None
        
        # read train files
        for filepath in filepaths:

            with open(filepath, 'rb') as fo:
                dict = cPickle.load(fo)

            data = dict['data']
            data = data.reshape(-1, 3, 32, 32)  # unflat
            data = np.transpose(data, (0, 2, 3, 1))  # channels at the end
            if colorspace == 'BGR':
                data = data[:, :, :, ::-1]  # rgb2bgr
            elif colorspace == "GRAY":
                data_gray = [cv2.cvtColor(data[idx], cv2.COLOR_BGR2GRAY) for idx in range(data.shape[0])]
                data = np.stack(data_gray, axis=0)
            
            labels = dict['labels']
            labels = np.stack(labels)
            
            if data_images is None:
                data_images = data
                data_labels = labels
            else:
                data_images = np.concatenate((data_images, data))
                data_labels = np.concatenate((data_labels, labels))
        
        return data_images, data_labels

class CifarCNNFeatures(object):

    def generate_random_batch(self, train=True, permute=True, batch_size=128, aug_level=0):
        """
        Take batch_size features from models.
    
        Args:
            train (bool): Train or test features?
            batch_size (int): Batch_size.
        Returns:
            (np.ndarray of size [B, E], np.ndarray of size [B, E]): Representative batch and it's labels.
        """
        if train:
            features_dir = os.path.join(DATASET_FEATURES_DIR + "-" + str(aug_level), TRAIN_DIR)
        else:
            features_dir = os.path.join(DATASET_FEATURES_DIR + "-" + str(aug_level), TEST_DIR)

        filepaths = [os.path.join(features_dir, filename) for filename in os.listdir(features_dir) if filename.endswith(".npy")]
        labels = [int(filename[filename.find("_") + 1:filename.find(".")]) for filename in os.listdir(features_dir) if filename.endswith(".npy")]

        # permute
        if permute:
            indices = np.arange(len(filepaths))
            np.random.shuffle(indices)
            filepaths = (np.array(filepaths)[indices]).tolist()
            labels = (np.array(labels)[indices]).tolist()

        # filepaths
        batch_features = []
        batch_labels = []
        
        for idx in range(len(filepaths)):
            
            # model
            batch_features.append(np.load(filepaths[idx]))
            batch_labels.append(labels[idx])
    
            # yield
            if idx % batch_size == batch_size - 1:
                ret = (np.stack(batch_features, axis=0), np.stack(batch_labels, axis=0))
                yield ret
                batch_features = []
                batch_labels = []

    def size(self, aug_level, train):

        if train:
            features_dir = os.path.join(DATASET_FEATURES_DIR + "-" + str(aug_level), TRAIN_DIR)
        else:
            features_dir = os.path.join(DATASET_FEATURES_DIR + "-" + str(aug_level), TEST_DIR)
    
        # Get labels
        filepaths = [os.path.join(features_dir, filename) for filename in os.listdir(features_dir) if filename.endswith(".npy")]            
        return len(filepaths)
