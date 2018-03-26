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

import sys
import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.svm import LinearSVC
from sklearn.manifold import TSNE
from utils import data_generator as gen

def _HOG(images):
    """
    Calc HOG features for grayscale images.

    Args:
        images (ndarray of size [images, 32, 32]): Grayscale images.

    Returns:
        (ndarray of size [images, features_no]): HOG features for each image.
    """

    WIN_SIZE = (32, 32)
    BLOCK_SIZE = (8, 8)
    BLOCK_STRIDE = (4, 4)
    CELL_SIZE = (4, 4)
    NBINS = 9

    hog_desriptor = cv2.HOGDescriptor(WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS)
    hog_features = [np.squeeze(hog_desriptor.compute(images[idx])) for idx in range(images.shape[0])]
    return np.stack(hog_features, axis=0)

def images_visualization(instances=10, scale=2.0):
    """
    Visualize some random images from the dataset.

    Args:
        instances (int): How many random images per class should I show?
        scale (float): How many times should I multiply canvas image?
    """
    
    BORDER = 1
    IMAGE_SIZE = 32
    CHANNELS = 3

    generator = gen.CifarImages('BGR')
    images = generator.get_random_images(train=True, instance_number=instances)
    canvas = np.ones(shape=(instances * 32 + (instances + 1) * BORDER,
                            instances * 32 + (instances + 1) * BORDER, CHANNELS), dtype=np.uint8) * 255
    for class_idx in range(len(images)):
        row = class_idx * IMAGE_SIZE + (class_idx + 1) * BORDER
        for instance_idx in range(instances):
            col = instance_idx * IMAGE_SIZE + (instance_idx + 1) * BORDER
            canvas[row:row + IMAGE_SIZE, col:col + IMAGE_SIZE, :] = images[class_idx][instance_idx]

    canvas = cv2.resize(src=canvas, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    cv2.imshow('examples', canvas)
    cv2.waitKey()
    cv2.destroyAllWindows()

def extract_HOG(generator, train, verbose=True):
    """
    Extract HOG features for specified dataset (train/test).

    Args:
        generator (Generator class object): Generator class object.
        train (bool): Am I working with train or test data?
        verbose (bool): Should I print some additional info?

    Returns:
        (ndarray of size [images, features_no], ndarray of size [images]) Features and labels. 
    """
    samples = generator.size(train=train)
    for images, labels in generator.generate_batch(train=train, batch_size=samples):
        start_time = time.time()
        features = _HOG(images)
        hog_time = time.time()
        if verbose:
            print "Features calculated in ", hog_time - start_time, " seconds"
        return features, labels

def decompose(features, labels):
    """
    Decompose features with some dimensionality reduction techniques.

    Args:
        features (ndarray of size [images, features_no]): Features of the dataset.
        labels (ndarray of size [images]): Labels of the dataset.
    """

    # PCA decomposition 
    start_time = time.time()
    pca = decomposition.PCA(n_components=2)
    pca.fit(features)
    features_pca = pca.transform(features)
    pca_time = time.time()
    print "PCA features calculated in ", pca_time - start_time, " seconds with variance ", pca.explained_variance_ratio_

    # t-SNE decomposition
    elems = 5000
    tsne = TSNE(n_components=2, verbose=True, perplexity=40, n_iter=300)
    features_tsne = tsne.fit_transform(features[:elems], labels[:elems])
    tsne_time = time.time()
    print "t-SNE features calculated in ", tsne_time - pca_time, " seconds "

    # plots
    plt.figure(figsize=(15, 15))
    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels)
    plt.figure(figsize=(15, 15))
    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels[:elems])
    plt.show()

def hparam_search(features_train, labels_train, features_test, labels_test):
    """
    Search best C param for SVM classifier but first reduce dimension of the features.

    Args:
        features_train (ndarray of size [images, features_no]): Features of the train dataset.
        labels_train (ndarray of size [images]): Labels of the train dataset.
        features_test (ndarray of size [images, features_no]): Features of the test dataset.
        labels_test (ndarray of size [images]): Labels of the test dataset.
    """

    VARIANCE = 0.60
    pca_hparam = decomposition.PCA(VARIANCE)
    pca_hparam.fit(features_train)
    features_hparam_train = pca_hparam.transform(features_train)

    print "Componenst with ", VARIANCE * 100, "% of variance: ", pca_hparam.n_components_
    for C in [0.001, 0.01, 0.1, 1.0, 1.2, 1.5, 2.0, 10.0]:
        classifier_svm = LinearSVC(C=C, verbose=False)
        classifier_svm.fit(features_hparam_train, labels_train)
        print "======= C:", C, "======="
        print "TRAIN SCORE = ", classifier_svm.score(features_hparam_train, labels_train)
        features_hparam_test = pca_hparam.transform(features_test)
        print "TEST  SCORE = ", classifier_svm.score(features_hparam_test, labels_test)

def classify(features_train, labels_train, features_test, labels_test, C, verbose):
    """
    Train SVM classifier and eval train and test scores.

    Args:
        features_train (ndarray of size [images, features_no]): Features of the train dataset.
        labels_train (ndarray of size [images]): Labels of the train dataset.
        features_test (ndarray of size [images, features_no]): Features of the test dataset.
        labels_test (ndarray of size [images]): Labels of the test dataset.
        C (float): C parameter of SVM classifier.
        verbose (bool): Should I print some additional info?
    """
    svm_time_start = time.time()
#    classifier_svm = LinearSVC(C=C, verbose=verbose, dual=False, max_iter=5000)
    classifier_svm = LinearSVC(C=C, verbose=verbose, dual=True, max_iter=1000)
    classifier_svm.fit(features_train, labels_train)
    svm_time_fit = time.time()
    print "SVM fit in ", svm_time_fit - svm_time_start, " seconds\n\n"
    print "TRAIN SCORE = ", classifier_svm.score(features_train, labels_train)
    print "TEST  SCORE = ", classifier_svm.score(features_test, labels_test)

def main(argv):

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--features", help="supported image features: hog, cnn", type=str)
    parser.add_argument("-a", "--augmentation_level", help="if cnn features were chosen, which dataset should I use?  (0, 1 or 2)", type=int)
    parser.add_argument('-i', '--images_vis', help='visualize class images', action='store_true')
    parser.add_argument('-s', '--hparam_search', help='search for best C param for SVM classifier', action='store_true')
    parser.add_argument('-t', '--train', help='train SVM classifier', action='store_true')
    parser.add_argument('-d', '--decompose', help='decompose features with PCA and t-SNE', action='store_true')
    parser.add_argument('-v', '--verbose', help='should I print some additional info?', action='store_true')
    parser.add_argument("-c", "--c_param", help="C param of the SVM classifier", type=float, default=1.0)
    args = vars(parser.parse_args())  

    ######################################################################################
    ################################ IMAGES VISUALIZATION ################################
    ######################################################################################

    if args['images_vis']:
        images_visualization()

    ######################################################################################
    ################################## Extract features ##################################
    ######################################################################################

    if args['features'] == 'hog':
        generator = gen.CifarImages('GRAY')
        features_train, labels_train = extract_HOG(generator, train=True, verbose=args['verbose'])
        features_test, labels_test = extract_HOG(generator, train=False, verbose=args['verbose'])
    elif args['features'] == 'cnn':
        if args['augmentation_level'] == None:
            raise ValueError("augmentation_level option should be used here")
            exit()
        generator = gen.CifarCNNFeatures()
        for data in generator.generate_random_batch(train=True, permute=False,
                                                    batch_size=generator.size(args['augmentation_level'], train=True),
                                                    aug_level=args['augmentation_level']):
            features_train = data[0]
            labels_train = data[1]
        for data in generator.generate_random_batch(train=False, permute=False,
                                                    batch_size=generator.size(args['augmentation_level'], train=False),
                                                    aug_level=args['augmentation_level']):
            features_test = data[0]
            labels_test = data[1]
    elif args['features'] == None:
        return
    else:
        raise ValueError("Features not supported..")
        exit()

    ######################################################################################
    ################################## PCA DECOMPOSITION #################################
    ######################################################################################

    if args['decompose']:
        decompose(features_train, labels_train)

    ######################################################################################
    #################################### HPARAM SEARCH ###################################
    ######################################################################################

    if args['hparam_search']:
        hparam_search(features_train, labels_train, features_test, labels_test)

    ######################################################################################
    ################################ FINAL CLASSIFICATION ################################
    ######################################################################################

    if args['train']:
        classify(features_train, labels_train, features_test, labels_test, args['c_param'], args['verbose'])


if __name__ == "__main__":
    main(sys.argv[1:])
