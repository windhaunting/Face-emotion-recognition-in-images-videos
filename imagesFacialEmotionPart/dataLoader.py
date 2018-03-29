#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 01:00:21 2018

@author: fubao
"""

from os.path import join
import numpy as np
import cv2
from common import *


class FerDatasetLoader(object):

  def __init__(self):
    pass

  def load_from_save(self):
    self._images      = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_IMAGES_FILENAME))
    self._labels      = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_LABELS_FILENAME))
    self._images_test = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_IMAGES_TEST_FILENAME))
    self._labels_test = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_LABELS_TEST_FILENAME))
    self._images      = self._images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    self._images_test = self._images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    self._labels      = self._labels.reshape([-1, len(EMOTIONS)])
    self._labels_test = self._labels.reshape([-1, len(EMOTIONS)])

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def images_test(self):
    return self._images_test

  @property
  def labels_test(self):
    return self._labels_test