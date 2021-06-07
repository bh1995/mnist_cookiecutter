# -*- coding: utf-8 -*-
"""train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1S18ewAG_5bFZvOeGfTdaphaZM0BcdUOg
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.optim as optim

# from data1 import *
# train_loader, test_loader = load_mnist() # this will download mnist data

# from model import *
# device = torch.device('cpu') 
# model = Net()
# model = model.to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

def train_function(model, train_loader, optimizer, n_epochs=10):
  # optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
  train_losses = []
  model.train()
  for epoch in tqdm(range(1, n_epochs + 1), desc='Epoch loop'):
    epoch_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      output = model(data)
      # loss = loss_function(output, target)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      epoch_loss.append(loss.item())
    train_losses.append(np.mean(epoch_loss))
    if epoch%5 == 0:
      # print('Epoch:', epoch, ', Loss:', train_losses[-1])
      print(f'Epoch {epoch}: train loss {train_losses[-1]}')
      plt.plot(np.arange(epoch), train_losses)
      # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
      #   epoch, batch_idx * len(data), len(train_loader.dataset),
      #   100. * batch_idx / len(train_loader), loss.item()))
      # train_counter.append(
      #   (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      # torch.save(model.state_dict(), '/results/model.pth')
      # torch.save(optimizer.state_dict(), '/results/optimizer.pth')
  # model.load_state_dict(best_model_wts)
  return model.eval(), train_losses