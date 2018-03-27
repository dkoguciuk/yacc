#!/usr/bin/python
# -*- coding: utf-8 -*

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
@note: Created on 24.03.2018
'''

import os
import sys
import cv2
import time
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from utils import data_generator as gen
import slim.nets.inception as inception_model
# import slim.preprocessing.inception_preprocessing as inception_preprocessing

# VARS
BATCH_SIZE = 100
TEST_DIR = 'test'
TRAIN_DIR = 'train'
FEATURES_DIR = 'cifar-10-batches-py-cnn-features'
INCEPTION_MODEL_PATH = 'inception/inception_v3.ckpt'

class InceptionV3(object):
    def __init__(self, sess):    
        # placeholder
        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
        # inception features extractor
        with tf.device('device:GPU:0'):
            with tf.contrib.slim.arg_scope(inception_model.inception_v3_arg_scope()):
                self.features_extractor, _ = inception_model.inception_v3(self.input_placeholder,
                                                                          num_classes=0, is_training=False)
        # init
        init_fn = tf.contrib.slim.assign_from_checkpoint_fn(INCEPTION_MODEL_PATH,
                                                            tf.contrib.slim.get_model_variables("InceptionV3"))
        init_fn(sess)

def augment_data_advanced(images):
    """
    Augment images with some adv techinques: colorshift, brightness shift, clahe, etc. 

    Args:
        images (list of ndarrays of shape [32, 32, 3]): Original images to be augmented.

    Returns:
        (list of ndarrays of shape [32, 32, 3]): Augmented images.
    """

    IMAGE_SIZE = 299
    
    # color shift +-10%
    images_color_shifted = []
    for idx in range(len(images)):
        image = images[idx].astype(np.int32)
        color_noise = np.random.randint(-0.1 * 256, 0.1 * 256, size=3, dtype=np.int32)
        image = np.clip(image + color_noise, 0, 255)
        images_color_shifted.append(image.astype(np.uint8))

    # brightness shift +-10%
    images_brightness_shifted = []
    for idx in range(len(images_color_shifted)):
        hsv = cv2.cvtColor(images_color_shifted[idx], cv2.COLOR_RGB2HSV).astype(np.int32)
        brightness_noise = np.random.randint(-0.1 * 256, 0.1 * 256, size=1, dtype=np.int32)
        hsv[:, :, 2] += brightness_noise
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        images_brightness_shifted.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    
    # clahe
    images_equalized = []
    clahe = cv2.createCLAHE(clipLimit=2.0)
    for idx in range(len(images_brightness_shifted)):
        lab = cv2.cvtColor(images_brightness_shifted[idx], cv2.COLOR_RGB2LAB)
        labs = cv2.split(lab)
        labs[0] = clahe.apply(labs[0])
        lab = cv2.merge(labs)
        images_equalized.append(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

    # return
    return images_equalized

def augment_data_basic(images):
    """
    Augment images with some basic techinques: cropping, translating, flipping and rotating.

    Args:
        images (list of ndarrays of shape [32, 32, 3]): Original images to be augmented.

    Returns:
        (list of ndarrays of shape [32, 32, 3]): Augmented images.
    """

    IMAGE_SIZE = 299
    
    # random crop without aspect ratio
    crop_ys_s = np.random.randint(0, int(round(0.1 * IMAGE_SIZE)) + 1, size=len(images))
    crop_ys_e = np.random.randint(int(round(0.9 * IMAGE_SIZE)), IMAGE_SIZE + 1, size=len(images))
    crop_xs_s = np.random.randint(0, int(round(0.1 * IMAGE_SIZE)) + 1, size=len(images))
    crop_xs_e = np.random.randint(int(round(0.9 * IMAGE_SIZE)), IMAGE_SIZE + 1, size=len(images))
    images_cropped = [images[idx][crop_ys_s[idx]:crop_ys_e[idx],
                                  crop_xs_s[idx]:crop_xs_e[idx], :].copy() for idx in range(len(images))]

    # resize to 299
    images_resized = [cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4) for image in images_cropped]

    # random translation
    images_translated = []
    translation_x = np.random.randint(int(round(-0.1 * IMAGE_SIZE)), int(round(0.1 * IMAGE_SIZE)) + 1, size=len(images_resized))
    translation_y = np.random.randint(int(round(-0.1 * IMAGE_SIZE)), int(round(0.1 * IMAGE_SIZE)) + 1, size=len(images_resized))
    for idx in range(len(images_resized)):
        M = np.float32([[1, 0, translation_x[idx]], [0, 1, translation_y[idx]]])
        images_translated.append(cv2.warpAffine(images_resized[idx], M, images_resized[idx].shape[:2]))

    # flip
    images_flipped = []
    flip = np.random.uniform(size=len(images_translated)) < 0.33
    for idx in range(len(images_translated)):
        if flip[idx]:
            images_flipped.append(cv2.flip(images_translated[idx], 1))
        else:
            images_flipped.append(images_translated[idx])

    # rotate
    images_rotated = []
    rotate_base = np.random.uniform(size=len(images_flipped))
    rotate_left = rotate_base < 0.33
    rotate_right = rotate_base > 0.66
    for idx in range(len(images_flipped)):
        if rotate_left[idx]:
            images_rotated.append(np.rot90(images_flipped[idx]))
        elif rotate_right[idx]:
            images_rotated.append(np.rot90(images_flipped[idx], 3))
        else:
            images_rotated.append(images_flipped[idx])

    # return
    return images_rotated

def calc_features(train, multiplications, augmentation, out_dir, inception, sess):
    """
    Calculate cnn features with inception v3.

    Args:
        train (bool): Am I working with train or test data?
        multiplications (int): How many times should I augment data (should be 1 for
            test dataset).
        augmentation (int): Augmentation level, either 0, 1 or 2.
        out_dir (str): Output directory, where cnn codes will be saved.
        inception (InceptionV3 object): Inception model.
        sess (Tensorflow session): Tensorflow session.
    """
    
    # counter
    instances_counter = { str(idx) : 0 for idx in range(10)}  
    
    # for every batch
    generator = gen.CifarImages('RGB')
    batches = generator.size(train) / BATCH_SIZE
    
    for _ in range(multiplications):
        for batch_images, batch_labels in tqdm(generator.generate_batch(train, batch_size=BATCH_SIZE), total=batches):
    
            # resize
            images_resized = [cv2.resize(batch_images[idx], (299, 299), interpolation=cv2.INTER_LANCZOS4)
                              for idx in range(batch_images.shape[0])]

            # augmentation
            if augmentation == 3:
                images_augmented = augment_data_advanced(images_resized)
            if augmentation == 2:
                temp = augment_data_advanced(images_resized)
                images_augmented = augment_data_basic(temp)
            if augmentation == 1:
                images_augmented = augment_data_basic(images_resized)
            else:
                images_augmented = images_resized
            
            # norm
            images_normalized = [cv2.normalize(src=image, dst=None, alpha=-1.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                                 for image in images_augmented]
            
            # to tf format 
            np_images_data = [np.asarray(image_normalized) for image_normalized in images_normalized]
            
            # stack
            tf_images_data = np.stack(np_images_data, axis=0)
            
            # calc features
            features = sess.run(inception.features_extractor, feed_dict={inception.input_placeholder : tf_images_data})
            features = np.squeeze(features)
    
            # save features
            for idx in range(len(batch_images)):
                class_idx = "%04d" % (instances_counter[str(batch_labels[idx])])
                filename = class_idx + "_" + str(batch_labels[idx]) + ".npy"
                filepath = os.path.join(out_dir, filename)
                np.save(filepath, features[idx])
                instances_counter[str(batch_labels[idx])] += 1

def main(argv):

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--augmentation_level", help="augment dataset: option 0, 1 or 2, 3", type=int, required=True)
    parser.add_argument("-m", "--multiplications", help="how many times should I augment training dataset?", type=int, required=True)
    args = vars(parser.parse_args())

    # assert aug level
    if args['augmentation_level'] != 0 and args['augmentation_level'] != 1 and args['augmentation_level'] != 2 and args['augmentation_level'] != 3:
        print "augmentation level can be one of: 0, 1, 2"
        exit()

    # assert features dir
    out_dir = FEATURES_DIR + "-" + str(args['augmentation_level'])
    if os.path.exists(out_dir):
        print "features dir exists!"
        exit()
    # create features dir
    os.mkdir(out_dir)
    for inner_dir in [TRAIN_DIR, TEST_DIR]:
        os.mkdir(os.path.join(out_dir, inner_dir))
    
    # tf session & graph
    sess = tf.InteractiveSession()
    graph = tf.Graph()
    graph.as_default()
    
    # InceptionV3
    inception = InceptionV3(sess)
     
    # train
    print "Calculating features for train dir..."
    calc_features(True, args['multiplications'], args['augmentation_level'], os.path.join(out_dir, TRAIN_DIR), inception, sess)
    
    # test
    print "Calculating features for train dir..."
    calc_features(False, 1, 0, os.path.join(out_dir, TEST_DIR), inception, sess)

if __name__ == "__main__":
    main(sys.argv[1:])
