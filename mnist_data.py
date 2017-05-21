#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 19:12:15 2017
Loads the MNIST data into training and testing sets
(X_train, Y_train) and (X_test, Y_test), respectively.

X_train and X_test are normalized matrices where "one row = one sample",
with values between -1 and 1 unless specified.

Y_train and Y_test are arrays/lists.

@author: bettmensch
"""

import numpy
import os
import struct
from array import array
import random

_allowed_modes = (
    # integer values in {0..255}
    'vanilla',

    # integer values in {0,1}
    # values set at 1 (instead of 0) with probability p = orig/255
    # as in Ruslan Salakhutdinov and Iain Murray's paper
    # 'On The Quantitative Analysis of Deep Belief Network' (2008)
    'randomly_binarized',

    # integer values in {0,1}
    # values set at 1 (instead of 0) if orig/255 > 0.5
    'rounded_binarized',
)

_allowed_return_types = (
    # default return type. Computationally more expensive.
    # Useful if numpy is not installed.
    'lists',

    # Numpy module will be dynamically loaded on demand.
    'numpy',
)

np = None

def _import_numpy():
    # will be called only when the numpy return type has been specifically
    # requested via the 'return_type' parameter in MNIST class' constructor.
    global np
    if np is None: # import only once
        try:
            import numpy as _np
        except ImportError as e:
            raise MNISTException(
                "need to have numpy installed to return numpy arrays."\
                +" Otherwise, please set return_type='lists' in constructor."
            )
        np = _np
    else:
        pass # was already previously imported
    return np

class MNISTException(Exception):
    pass

class MNIST(object):
    def __init__(self, path='.', mode='vanilla', return_type='lists'):
        self.path = path

        assert mode in _allowed_modes, \
            "selected mode '{}' not in {}".format(mode,_allowed_modes)

        self._mode = mode

        assert return_type in _allowed_return_types, \
            "selected return_type '{}' not in {}".format(
                return_type,
                _allowed_return_types
            )

        self._return_type = return_type

        self.test_img_fname = 'mnist_testing_images'
        self.test_lbl_fname = 'mnist_testing_labels'

        self.train_img_fname = 'mnist_training_images'
        self.train_lbl_fname = 'mnist_training_labels'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    @property # read only because set only once, via constructor
    def mode(self):
        return self._mode

    @property # read only because set only once, via constructor
    def return_type(self):
        return self._return_type

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname))

        self.test_images = self.process_images(ims)
        self.test_labels = self.process_labels(labels)

        return self.test_images, self.test_labels

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname))

        self.train_images = self.process_images(ims)
        self.train_labels = self.process_labels(labels)

        return self.train_images, self.train_labels

    def process_images(self, images):
        if self.return_type is 'lists':
            return self.process_images_to_lists(images)
        elif self.return_type is 'numpy':
            return self.process_images_to_numpy(images)
        else:
            raise MNISTException("unknown return_type '{}'".format(self.return_type))

    def process_labels(self, labels):
        if self.return_type is 'lists':
            return labels
        elif self.return_type is 'numpy':
            _np = _import_numpy()
            return _np.array(labels)
        else:
            raise MNISTException("unknown return_type '{}'".format(self.return_type))

    def process_images_to_numpy(self,images):
        _np = _import_numpy()

        images_np = _np.array(images)

        if self.mode == 'vanilla':
            pass # no processing, return them vanilla

        elif self.mode == 'randomly_binarized':
            r = _np.random.random(images_np.shape)
            images_np = (r <= ( images_np / 255)).astype('int') # bool to 0/1

        elif self.mode == 'rounded_binarized':
            images_np = ((images_np / 255) > 0.5).astype('int') # bool to 0/1

        else:
            raise MNISTException("unknown mode '{}'".format(self.mode))

        return images_np

    def process_images_to_lists(self,images):
        if self.mode == 'vanilla':
            pass # no processing, return them vanilla

        elif self.mode == 'randomly_binarized':
            for i in range(len(images)):
                for j in range(len(images[i])):
                    pixel = images[i][j]
                    images[i][j] = int(random.random() <= pixel/255) # bool to 0/1

        elif self.mode == 'rounded_binarized':
            for i in range(len(images)):
                for j in range(len(images[i])):
                    pixel = images[i][j]
                    images[i][j] = int(pixel/255 > 0.5) # bool to 0/1
        else:
            raise MNISTException("unknown mode '{}'".format(self.mode))

        return images

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels

    @classmethod
    def display(cls, img, width=28, threshold=200):
        render = ''
        for i in range(len(img)):
            if i % width == 0:
                render += '\n'
            if img[i] > threshold:
                render += '@'
            else:
                render += '.'
        return render

class normalizer(object):
    """Takes a matrix where "one row = one sample" and normalizes entries
    based on maximum-minimum value range over all entries.
    Returns normalized (all entries in the range (-1,1)) matrix."""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
        
    def fit(self, X):
        self.mean = numpy.mean(X)
        self.std = numpy.std(X)
        self.fitted = True
        
    def fit_transform(self, X):
        self.fit(X)
        self.fitted = True
        
        return (X - self.mean) / self.std
    
    def transform(self, X):
        if self.fitted:
            return (X - self.mean) / self.std
    
def get_mnist_data(factor = 1):
    mndata = MNIST('./mnist_data')

    X_train_init, y_train = mndata.load_training()
    X_test_init, y_test = mndata.load_testing()
    
    y_train = numpy.array([y_i for y_i in y_train])
    y_test = numpy.array([y_i for y_i in y_test])
    
    norm = normalizer()

    X_train_mat = numpy.matrix([x_sample for x_sample in X_train_init])
    X_train = norm.fit_transform(X_train_mat)

    X_test_mat = numpy.matrix([x_sample for x_sample in X_test_init])
    X_test = norm.transform(X_test_mat)
        
    return X_train, y_train, X_test, y_test