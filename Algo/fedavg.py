import torch
from torch import nn
import copy
import numpy as np
from Algo.communication import communication
from util.train_eval import train, evaluate
from util.print_info import print_epoch_end

class fedavg(nn.Module):
    def __init__(self, server_model, scenario, loss_fun, fed_method='fedavg'):
        super(fedavg, self).__init__()
        self.server_model = server_model
        self.server_model.eval()
        self.scenario = scenario
        self.client_model = self.scenario.init_client_models(server_model)
        self.loss_fun = loss_fun
        self.fed_method = fed_method
        self.history = [list() for _ in range(scenario.n_clients)]
        self.selected_client_index = np.arange(self.scenario.n_clients_each_round)
        self.selected_distributed_dataloaders = self.scenario.distributed_dataloaders[:self.scenario.n_clients_each_round]
        self.selected_client_weights = self.scenario.all_client_weights[:self.scenario.n_clients_each_round]

    '''def fit_in_one_epoch(self, idx, epochs, train_opt_func, optimizer, print_output):
        for epoch in range(epochs):
            self.client_model[idx].train()'''

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
                l, t, a = train(self.client_model[i], self.selected_distributed_dataloaders[i], optimizer, self.loss_fun)
                if print_output:
                    print_epoch_end(epoch, l, t, a, output_file)

    def server_aggre(self):
        self.server_model, self.client_model = communication(self.server_model, self.client_model,
                                                self.selected_client_weights, self.fed_method)

    def client_eval(self, testloader):
        for i in range(self.scenario.n_clients_each_round):
            train_loss, train_acc = evaluate(self.client_model[i], testloader, self.loss_fun)
            print(f'Client_{self.selected_client_index[i]+1}: Train_loss: {train_loss}; Accuracy: {train_acc}')

    def server_eval(self, testloader, nround, output_file):
        eval_loss, eval_acc = evaluate(self.server_model, testloader, self.loss_fun)
        print(f'Comm_round_{nround+1} Server model: Eval_loss: {eval_loss}; Accuracy: {eval_acc}', file=output_file)
        return eval_loss, eval_acc