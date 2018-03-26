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
import cv2
import tarfile
import cPickle
import requests
import numpy as np
from tqdm import tqdm
from utils import data_generator as gen

CIFAR_DIR = 'cifar-10-batches-py'
CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
INCEPTION_DIR = 'inception'
INCEPTION_URL = 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz'

def download_inception_ckpt(url, path='inception'):
    """
    Download inception model archive from specified url and save it as filename.

    Args:
        url (str): URL to the inception model archive.
        filename (str): How should I name local ckpt model?

    Returns:
        (str): Local model filename.
    """
    CHUNK_SIZE = 1024

    archive_name = url[url.rfind('/') + 1:]
    r_head = requests.head(url, stream=True)
    filesize = int(r_head.headers['Content-length'])
    
    r_get = requests.get(url, stream=True)
    with open(archive_name, 'wb') as f:
        for chunk in tqdm(r_get.iter_content(chunk_size=CHUNK_SIZE), total=filesize / CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    if (archive_name.endswith("tar.gz")):
        tar = tarfile.open(archive_name, "r:gz")
        tar.extractall(path=path)
        tar.close()        
        os.remove(archive_name)
    else:
        raise ValueError("Sorry, don't know this archive extension.")

def download_cifar(url, filename=None):
    """
    Download cifar archive from specified url and save it as filename.

    Args:
        url (str): URL to the cifar archive.
        filename (str): How should I name local archive? Will be extracted
            from url if not provided.

    Returns:
        (str): Local archive filename.
    """
    
    CHUNK_SIZE = 1024
    
    if filename == None:
        filename = url[url.rfind('/') + 1:]

    r_head = requests.head(url, stream=True)
    filesize = int(r_head.headers['Content-length'])
    
    r_get = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in tqdm(r_get.iter_content(chunk_size=CHUNK_SIZE), total=filesize / CHUNK_SIZE):
            if chunk:
                f.write(chunk)
    
    return filename
def extract_cifar(filename, delete_archive=True):
    """
    Extract cifar archive with specified name.

    Args:
        filename (str): Local archive filename.
        delete_archive (bool): Should I delete archive after extracting?
    """
    if (filename.endswith("tar.gz")):
        tar = tarfile.open(filename, "r:gz")
        tar.extractall()
        tar.close()
        
        if not os.path.exists(CIFAR_DIR):
            raise ValueError("Extracting error..")
        
        if delete_archive:
            os.remove(filename)
    else:
        raise ValueError("Sorry, don't know this archive extension.")

def main(argv):

    if not os.path.exists(INCEPTION_DIR):

        print "Downloading incpetion model..."
        download_inception_ckpt(INCEPTION_URL)

    else:
        print "Inception model already downloaded!"

    if not os.path.exists(CIFAR_DIR):
        
        print "Downloading dataset..."
        filename = download_cifar(CIFAR_URL, filename=None)
    
        print "Extracting dataset..."
        extract_cifar(filename, delete_archive=True)
        
        print "All done!"
    
    else:
        print "Dataset already downloaded!"
    

if __name__ == "__main__":
    main(sys.argv[1:])
