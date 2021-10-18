import torch
import torch.nn.functional as F
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from preprocess import CompDataset
from preprocess import get_user_data

class Worker(object):
    def __init__(self, user_idx):
        self.user_idx = user_idx
        self.data,self.edges = get_user_data(self.user_idx)  # The worker can only access its own data
        self.ps_info = {}

    def preprocess_worker_data(self):
        self.data = self.data[self.data['class']!=2]
        x = self.data.iloc[:, 2:]
        x = x.reset_index(drop=True)
        x = x.to_numpy().astype(np.float32)
        y = self.data['class']
        y = y.reset_index(drop=True)
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        self.data = (x,y)

    def round_data(self, n_round, n_round_samples=-1):
        """Generate data for user of user_idx at round n_round.

        Args:
            n_round: int, round number
            n_round_samples: int, the number of samples this round
        """

        if n_round_samples == -1:
            return self.data

        n_samples = len(self.data[1])
        choices = np.random.choice(n_samples, min(n_samples, n_round_samples))

        return self.data[0][choices], self.data[1][choices]
    
    def receive_server_info(self, info): # receive info from PS
        self.ps_info = info
    
    def process_mean_round_train_acc(self): # process the "mean_round_train_acc" info from server
        mean_round_train_acc = self.ps_info["mean_round_train_acc"]
        # You can go on to do more processing if needed

    def user_round_train(self, model, device, n_round,  batch_size, n_round_samples=-1, debug=False):

        X,Y = self.round_data(n_round, n_round_samples)
        data = CompDataset(X=X, Y=Y)
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
        )

        model.train()

        correct = 0
        prediction = []
        real = []
        total_loss = 0
        model = model.to(device)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # import ipdb
            # ipdb.set_trace()
            # print(data.shape, target.shape)
            output = model(data)
            loss = F.nll_loss(output, target)
            total_loss += loss
            loss.backward()
            pred = output.argmax(
                dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            prediction.extend(pred.reshape(-1).tolist())
            real.extend(target.reshape(-1).tolist())

        grads = {'n_samples': data.shape[0], 'named_grads': {}}
        for name, param in model.named_parameters():
            grads['named_grads'][name] = param.grad.detach().cpu().numpy()
        
        worker_info = {}
        worker_info["train_acc"] = correct / len(train_loader.dataset)

        if debug:
            print('Training Loss: {:<10.2f}, accuracy: {:<8.2f}'.format(
                total_loss, 100. * correct / len(train_loader.dataset)))

        return grads, worker_info
