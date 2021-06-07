# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:02:13 2021

@author: bjorn

script to perfrom predictions
"""

import torch
import matplotlib.pyplot as plt


# load model and weights
device = torch.device('cpu') 
from model import *
model = Net()
model = model.to(device)
load_model_path = 'C:/Users/bjorn/OneDrive/Dokument/University/DTU/Machine Learning Operations 21/mnist_cookiecutter/models/model_1.pth'
model.load_state_dict(torch.load(load_model_path))
model = model.eval()
# load the user defined data from external folder
from src.data.prediction_data import *
data_path = 'C:/Users/bjorn/OneDrive/Dokument/University/DTU/Machine Learning Operations 21/mnist_cookiecutter/data/external/'
data_loader = load_prediction_data(data_path, show_image=False)
# perform predictions
predictions = []
model.eval()
with torch.no_grad():
  for data in data_loader:
    im = data[0]  
    data = data[0].to(device)
    # print(type(data))
    # print(data.shape)
    output = model(data[None,:,:,:])
    pred = output.data.max(1, keepdim=True)[1].cpu().numpy()[0][0]
    plt.imshow(im[0,:,:].cpu().numpy(), cmap='Greys')
    plt.title('Model prediction:'+str(pred))
    plt.show()
    
    print('Prediction: ', pred)
    predictions.append(pred)


  
  




