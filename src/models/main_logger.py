# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 11:37:16 2021

@author: bjorn

main script with logger capabilites
"""


# from google.colab import files
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

# C:\Users\bjorn\OneDrive\Dokument\University\DTU\Machine Learning Operations 21\mnist_cookiecutter
sys.path.append("C:/Users/bjorn/OneDrive/Dokument/University/DTU/Machine Learning Operations 21/mnist_cookiecutter")  # need to add path

# load data
from src.data.data1 import *

(
    train_loader,
    test_loader,
) = load_mnist()  # this will download mnist data, or load it if already downloaded
# load model
device = torch.device("cpu")
from src.models.model import *

model = Net()
model = model.to(device)
# load trainer and return trained model
from src.models.train import train_function

# loss_function = F.nll_loss()
config = dict(
    epochs=30,
    classes=10,
    kernels=[16, 32],
    batch_size=256,
    learning_rate=0.000,
    dataset="MNIST",
    architecture="CNN")
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
model, train_losses = train_function(model, train_loader, optimizer, n_epochs=30, logger=True, config=None)

# evaluate model on test set
# not sure why, but test.py does not work, file must be named something else
from src.models.test1 import test_function

test_losses = test_function(model, test_loader)
# after 50 epochs -> Test set: Avg. loss: 0.1075, Accuracy: 9667/10000 (97%)
# after 30 epichs -> Test set: Avg. loss: 0.3469, Accuracy: 9082/10000 (91%)
# save test loss values
import pickle

name = "C:/Users/bjorn/OneDrive/Dokument/University/DTU/Machine Learning Operations 21/mnist_cookiecutter/reports/test_losses_1.csv"
open_file = open(name, "wb")
pickle.dump(test_losses, open_file)
open_file.close()
# save trained model weights for later evaluation
# 'C:/Users/bjorn/OneDrive/Dokument/University/DTU/Machine Learning Operations 21/mnist_cookiecutter/models/model_1.pth'
save_model_path = "./models/model_2.pth"
torch.save(model.state_dict(), save_model_path)





