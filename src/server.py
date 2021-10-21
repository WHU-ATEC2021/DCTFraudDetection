from datetime import datetime
import os
import shutil
import unittest
import pickle

import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F
from context import FederatedSGD
from context import PytorchModel
from learning_model import FLModel
from preprocess import get_test_data

class ParameterServer(object):
    def __init__(self, init_model_path, testworkdir, resultdir):
        self.round = 0
        self.rounds_info = {}
        self.rounds_model_path = {}
        self.worker_info = {}
        self.current_round_grads = []
        self.init_model_path = init_model_path
        self.aggr = FederatedSGD(
            model=PytorchModel(torch=torch,
                               model_class=FLModel,
                               init_model_path=self.init_model_path,
                               optim_name='Adam'),
            framework='pytorch',
        )
        self.testworkdir = testworkdir
        self.RESULT_DIR = resultdir
        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)
        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)
   
        self.test_data,self.test_edges = get_test_data()
        self.preprocess_test_data()

        self.round_train_acc= []
    
    def preprocess_test_data(self):
        self.predict_data = self.test_data[self.test_data['class']==3] # to be predicted
        self.predict_data_txId = self.predict_data[['txId','Timestep']]
        x = self.predict_data.iloc[:, 3:]
        x = x.reset_index(drop=True)
        x = x.to_numpy().astype(np.float32)
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        self.predict_data = x

    def get_latest_model(self):
        if not self.rounds_model_path:
            return self.init_model_path

        if self.round in self.rounds_model_path:
            return self.rounds_model_path[self.round]

        return self.rounds_model_path[self.round - 1]

    def receive_grads_info(self, grads): # receive grads info from worker
        self.current_round_grads.append(grads)
    
    def receive_worker_info(self, info): # receive worker info from worker
        self.worker_info = info
    
    def process_round_train_acc(self): # process the "round_train_acc" info from worker
        self.round_train_acc.append(self.worker_info["train_acc"])
    
    def print_round_train_acc(self):
        mean_round_train_acc = np.mean(self.round_train_acc) * 100
        print("\nMean_round_train_acc: ", "%.2f%%" % (mean_round_train_acc))
        self.round_train_acc = []
        return {"mean_round_train_acc":mean_round_train_acc
               }

    def aggregate(self):
        self.aggr(self.current_round_grads)

        path = os.path.join(self.testworkdir,
                            'round-{round}-model.md'.format(round=self.round))
        self.rounds_model_path[self.round] = path
        if (self.round - 1) in self.rounds_model_path:
            if os.path.exists(self.rounds_model_path[self.round - 1]):
                os.remove(self.rounds_model_path[self.round - 1])

        info = self.aggr.save_model(path=path)

        self.round += 1
        self.current_round_grads = []

        return info
    
    def save_prediction(self, predition):

        predition.to_csv(os.path.join(self.RESULT_DIR, 'result.csv'),index=0)
    
    def save_model(self,model):

        with open(os.path.join(self.RESULT_DIR, 'model.pkl'), 'wb') as fout:
            pickle.dump(model,fout)

    def save_testdata_prediction(self, model, device, test_batch_size):
        loader = torch.utils.data.DataLoader(
            self.predict_data,
            batch_size=test_batch_size,
            shuffle=False,
        )
        prediction = []
        model.eval()
        with torch.no_grad():
            for data in loader:
                pred = model(data.to(device)).argmax(dim=1, keepdim=True)
                prediction.extend(pred.reshape(-1).tolist())
        self.predict_data_txId['prediction'] = prediction

        
        self.save_prediction(self.predict_data_txId)
        self.save_model(model)

