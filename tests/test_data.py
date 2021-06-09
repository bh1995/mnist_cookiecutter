# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 06:40:47 2021

@author: bjorn

Test script for all tests used in project
"""

# test dataset

import pytest
import sys

# C:\Users\bjorn\OneDrive\Dokument\University\DTU\Machine Learning Operations 21\mnist_cookiecutter
sys.path.append("C:/Users/bjorn/OneDrive/Dokument/University/DTU/Machine Learning Operations 21/mnist_cookiecutter")  # need to add path
from src.data.data1 import *

def test_data():
    # load data
    # from src.data.data1 import *
    
    (
        train_loader,
        test_loader,
    ) = load_mnist()  # this will download mnist data, or load it if already downloaded
    batch_idx, (data, target) = next(iter(enumerate(train_loader)))
    # print(data.shape, target.shape)
    assert data.shape[0] == target.shape[0]
    assert data.shape[2] and data.shape[3] == 28


