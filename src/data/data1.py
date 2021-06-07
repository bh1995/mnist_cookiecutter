# -*- coding: utf-8 -*-
"""Untitled7.ipynb
make sure sys.path.append('./Machine Learning Operations 21/mnist_cookiecutter') 
is used before anything in main script	
"""

import torch
import torchvision

# from torchvision.datasets import MNIST


def load_mnist():
    # https://stackoverflow.com/questions/66646604/http-error-503-service-unavailable-when-trying-to-download-mnist-data
    # For some reason LeCun's website gives an error so here's another way to get the dataset.
    # !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
    # !tar -zxvf MNIST.tar.gz
    batch_size_train = 256
    batch_size_test = 1000
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    # C:/Users/bjorn/OneDrive/Dokument/University/DTU/Machine Learning Operations 21/mnist_cookiecutter/data/processed
    save_path = "./mnist_cookiecutter/data/"  # make sure correct path is initialized
    # train data
    train_data = torchvision.datasets.MNIST(
        root=save_path, train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size_train, shuffle=True
    )
    print("train_loader finished")
    # test data
    test_data = torchvision.datasets.MNIST(
        root=save_path, train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size_test, shuffle=True
    )
    print("test_loader finished")

    return train_loader, test_loader
