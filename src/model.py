from datetime import datetime
import os
import shutil
import unittest

import random

import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F

from context import FederatedSGD
from context import PytorchModel
from learning_model import FLModel
from worker import Worker
from server import ParameterServer


class FedSGDTestSuit(unittest.TestCase):
    RESULT_DIR = 'result'
    N_VALIDATION = 10000
    TEST_BASE_DIR = '/tmp/'

    def setUp(self):
        self.seed = 1024
        self.use_cuda = False
        self.batch_size = 64
        self.test_batch_size = 1000
        self.lr = 0.001
        self.n_max_rounds = 200
        self.log_interval = 20
        self.n_round_samples = 1600
        self.testbase = self.TEST_BASE_DIR
        self.n_users = 40
        self.testworkdir = os.path.join(self.testbase, 'competetion-test')

        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)

        self.init_model_path = os.path.join(self.testworkdir, 'init_model.md')
        self.set_seed(self.seed)

        if not os.path.exists(self.init_model_path):
            torch.save(FLModel().state_dict(), self.init_model_path)
        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

        self.ps = ParameterServer(init_model_path=self.init_model_path,
                                  testworkdir=self.testworkdir,resultdir=self.RESULT_DIR)
        
        self.workers = []
        for u in range(0, self.n_users):
            self.workers.append(Worker(user_idx=u))
            
    @staticmethod
    def set_seed(seed_val: int) -> None:
        print(f"using seed: {seed_val}")
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        random.seed(seed_val)
        os.environ['PYTHONHASHSEED'] = str(seed_val)

    def _clear(self):
        shutil.rmtree(self.testworkdir)

    def tearDown(self):
        self._clear()

    def test_federated_SGD(self):
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if self.use_cuda else "cpu")

        # 初始化每个workers的数据
        for u in range(0, self.n_users):
            self.workers[u].preprocess_worker_data()

        training_start = datetime.now()
        model = None
        # 每轮的流程包括
        # 1. (从Server处)获取上一轮的模型路径
        # 2. 对于每一个Worker节点, 进行以下操作
        #   2.1 加载上一轮模型
        #   2.2 每个Worker节点根据自己的数据计算梯度信息, 准确率信息
        #   2.3 Worker节点将相关信息传递给Server节点
        # 3. Server节点对地图信息进行聚合
        for r in range(1, self.n_max_rounds + 1):
            path = self.ps.get_latest_model()
            start = datetime.now()
            for u in range(0, self.n_users):
                model = FLModel()
                model.load_state_dict(torch.load(path))
                model = model.to(device)
                grads, worker_info = self.workers[u].user_round_train(model=model, device=device, n_round=r, batch_size=self.batch_size,
                                                                      n_round_samples=self.n_round_samples)

                self.ps.receive_grads_info(grads=grads)
                # The transfer of information from the worker to the server requires a call to the "ps.receive_worker_info"
                self.ps.receive_worker_info(worker_info)
                self.ps.process_round_train_acc()

            self.ps.aggregate()
            print('\nRound {} cost: {}, total training cost: {}'.format(
                r,
                datetime.now() - start,
                datetime.now() - training_start,
            ))

            if model is not None and r % self.log_interval == 0:
                # print average train acc and return
                server_info = self.ps.print_round_train_acc()
                # transport average train acc to each worker
                for u in range(0, self.n_users):
                    # The transfer of information from the server to the worker requires a call to the "worker.receive_server_info"
                    self.workers[u].receive_server_info(server_info)
                    # workers do processing
                    self.workers[u].process_mean_round_train_acc()

                self.ps.save_testdata_prediction(
                    model=model, device=device, test_batch_size=self.test_batch_size)

        if model is not None:
            self.ps.save_testdata_prediction(
                model=model, device=device, test_batch_size=self.test_batch_size)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(FedSGDTestSuit('test_federated_SGD'))
    return suite


def main():
    runner = unittest.TextTestRunner()
    runner.run(suite())


if __name__ == '__main__':
    main()
