# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:18:42 2021

@author: bjorn

script to load and process images from outside mnist
"""

import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize


def load_prediction_data(path, show_image=False):
    size = (28, 28)
    images = []
    image_names = [f.name for f in os.scandir(path)]
    if ".gitkeep" in image_names:
        image_names.remove(".gitkeep")
    # load images
    for image_name in image_names:
        # im = io.imread(os.path.join(path+'/image', image_name))
        im = Image.open(os.path.join(path, image_name)).convert("1")
        im = np.array(im, np.float32)
        im = resize(im, size, anti_aliasing=True)
        # if show_image:
        #     plt.imshow(im)
        #     plt.show()
        # im = np.resize(im, size)
        images.append(im)
        if show_image:
            plt.imshow(im)
            plt.show()

    # put data into tensor->dataset->data_loader
    data_loader = torch.utils.data.DataLoader(
        list(zip(images)), batch_size=1, shuffle=True
    )

    return data_loader
