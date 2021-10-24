import torch
import torch.nn.functional as F
import os
import math
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
        self._data,self.edges = get_user_data(self.user_idx)  # The worker can only access its own data
        self.ps_info = {}

    def preprocess_worker_data(self):
        self.data = self._data[self._data['class']!=2]
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
        
    def _ssl_data_augment(self, model) -> None:
        device = "cpu"
        unknown_data = self._data[self._data['class'] == 2]
        print(f"original data shape: {self.data[0].shape}")
        x = unknown_data.iloc[:, 2:]
        x = x.reset_index(drop=True)
        x = x.to_numpy().astype(np.float32)
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        
        loader = torch.utils.data.DataLoader(
            x,
            batch_size=64,
            shuffle=False,
        )
        x_list = []
        y_list = []
        model.eval()
        with torch.no_grad():
            for data in loader:
                pred_y = model(data.to(device))
                mask_y = pred_y.gt(math.log(0.98))
                mask_y = mask_y[:, 0] | mask_y[:, 1]
                pred_y = torch.argmax(pred_y, dim=1)
                selected_y = pred_y[mask_y]
                selected_x = data[mask_y, :]
                
                mask_y0 = selected_y.eq(0)
                selected_y = selected_y[mask_y0]
                selected_x = selected_x[mask_y0, :]
                
                x_list.append(selected_x.numpy())
                y_list.append(selected_y.numpy())
                
        xs = np.concatenate(x_list, axis=0)
        ys = np.concatenate(y_list, axis=0)
        
        
        nx = np.concatenate([self.data[0], xs], axis=0)
        ny = np.concatenate([self.data[1], ys], axis=0)
        self.data = (nx, ny)
        
        print(f"augment data shape: {self.data[0].shape}")
        print(f"user: {self.user_idx} augment data done")
        

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
