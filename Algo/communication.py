import torch
import copy
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

def communication(server_model, models, client_weights, fed_method='fedavg', total_n_clietns=100, device='cuda'):
    client_num = len(models)
    sum_weights = torch.tensor(np.sum(client_weights), dtype=torch.float, device=device)
    with torch.no_grad():
        if fed_method == 'fedbn':
            for key in server_model.trainable_keys:   #server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    temp = torch.div(temp, sum_weights)
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif fed_method == 'scaffold':
            #aggregated_delta_y = {}
            #aggregated_delta_control = {}
            global_lr = 1.0
            for key in server_model.trainable_keys:   #server_model.state_dict().keys():
                temp_delta_y = torch.zeros_like(server_model.delta_y[key], dtype=torch.float32).to(device)
                temp_delta_control = torch.zeros_like(server_model.delta_control[key], dtype=torch.float32).to(device)
                #print(temp_delta_control.shape)
                for client_idx in range(client_num):
                    #print(models[client_idx].delta_control[key].shape)
                    temp_delta_y += models[client_idx].delta_y[key].data
                    temp_delta_control += models[client_idx].delta_control[key].data
                temp_delta_y = torch.div(temp_delta_y, client_num)
                #temp_delta_control = torch.div(temp_delta_control, client_num)
                with torch.no_grad():
                    #print('***')
                    temp_delta_y = temp_delta_y*global_lr + server_model.state_dict()[key].data #server model's lr =1
                    server_model.state_dict()[key].data.copy_(temp_delta_y)
                    server_model.control[key] += temp_delta_control*(1/total_n_clietns)
                    #hard_code total_num_clients=100
                for client_idx in range(client_num):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif fed_method == 'simple_gmm_prompt': 
            for key in server_model.trainable_keys:
                if 'prompt' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    temp = torch.div(temp, sum_weights)
                else:
                    temp = list()
                    prompt_length = server_model.state_dict()[key].shape[1]
                    for client_idx in range(client_num):
                        temp.append(copy.deepcopy(models[client_idx].state_dict()[key].to('cpu').squeeze().numpy()))
                    temp = np.concatenate(temp)
                    gmm = BayesianGaussianMixture(n_components=prompt_length, n_init=prompt_length).fit(temp)
                    temp = torch.from_numpy(gmm.means_).to(device).type(torch.float32)
                    del gmm
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.trainable_keys:   #server_model.state_dict().keys():
                '''if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].dataa.copy_(models[0].state_dict()[key])
                else:'''
                temp = torch.zeros_like(server_model.state_dict()[key])
                for client_idx in range(client_num):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                temp = torch.div(temp, sum_weights)
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models