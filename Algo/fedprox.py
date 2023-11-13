import torch
from torch import nn
import copy
import numpy as np
from Algo.fedavg import fedavg
from util.train_eval import train, evaluate, train_prox
from util.print_info import print_epoch_end

class fedprox(fedavg):
    def __init__(self, server_model, scenario, loss_fun, mu, fed_method='fedprox'):
        super(fedprox, self).__init__(server_model, scenario, loss_fun, fed_method)
        self.mu = mu

    def client_train(self, comm_round, epochs, lr, output_file, opt_func=torch.optim.Adam, print_output=False):
        if self.scenario.type == 'cross_devices':
            self.selected_client_index, self.selected_distributed_dataloaders, self.selected_client_weights \
            = self.scenario.cross_devices_random_selecting()
        for i in range(self.scenario.n_clients_each_round):
            torch.cuda.empty_cache()
            optimizer = opt_func(filter(lambda p : p.requires_grad, self.client_model[i].parameters()), lr,
                                 betas=(0.9, 0.98), eps=1e-6)
            if print_output:
                print(f'------------Client_{self.selected_client_index[i]+1} start local trainig------------',
                      file=output_file)
            for epoch in range(epochs):
                self.client_model[i].train()
                if comm_round > 0:
                    l, t, a = train_prox(self.client_model[i], self.server_model,
                                         self.selected_distributed_dataloaders[i],
                                         optimizer, self.loss_fun, self.mu)
                else:
                    l, t, a = train(self.client_model[i], self.selected_distributed_dataloaders[i],
                                    optimizer, self.loss_fun)
                if print_output:
                    print_epoch_end(epoch, l, t, a, output_file)