import os
import argparse
import torch
from torch import nn
import torch.utils.data as Data
import numpy as np
import torchvision
import pytz
from datetime import datetime
from Algo.get_algos import get_algorithm
from data_util.DataTransformBuilder import build_data_transform
from data_util.DataDistributer import DataPartitioner
from Scenario import FL_scenario
from util.TOdevice import to_device
from Models.Prompted_models import Prompted_ViT_B32
from Models.pFedPG_model import client_prompted_vit_b32, BaseHeadsForLocal, prompt_generator
from util.train_eval import evaluate_pFedPG, train_eval_pFedPG, evaluate_all_pFedPG
from util.saving_tools import save_eval_npy_file, save_model_trainable_part, save_pfedpg_baseHeads

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='Algorithm to choose: [fedavg | fedprox | fedopt | scaffold | fedprox_gmm | pfedpg]')
    #parser.add_argument('--scenario', type=str, default='cross_devices',
                        #help='Federated Learning Scenario to choose: [cross_devices | cross_silo]')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Dataset to choose: [cifar10 | cifar100]')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='[cuda | cpu]')
    parser.add_argument('--batch', type=int, default=64, 
                        help='batch size')
    parser.add_argument('--comms', type=int, default=100, 
                        help='communication rounds')
    parser.add_argument('--local_eps', type=int, default=5,
                        help='number of epochs in local clients training')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--n_clients', type=int, default=100,
                        help='the number of clients')
    parser.add_argument('--n_sampled_clients', type=int, default=10,
                        help='the number of sampled clients per round')
    parser.add_argument('--data_distribution', type=str, default='non_iid_dirichlet',
                        help='data split way to choose: [non_iid_dirichlet | manual_extreme_heterogeneity]')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='the level of non-iid data split')
    parser.add_argument('--n_dominated_class', type=int, default=1,
                        help='number of dominated class when applying manual_heterogeneity')
    parser.add_argument('--prompt_method', type=str, default='shallow',
                        help='[shallow | deep]')
    parser.add_argument('--n_tokens', type=int, default=10,
                        help='number of tokens in prompt')
    
    # algorithm-specific parameters
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='The hyper parameter for fedprox')
    parser.add_argument('--fedopt_global_lr', type=float, default=5e-5,
                        help='Global model updating learning rate for fedopt')
    parser.add_argument('--prompt_lr', type=float, default=1e-3,
                        help='Prompt generaotr updating learning rate for pFedPG')
    parser.add_argument('--save_model', type=bool, default=False,
                        help='Save the trained model in the last epoch')
    args = parser.parse_args()
    print(args.device)
    #check if cross_devices or corss_silo
    if args.n_clients > args.n_sampled_clients:
        scenrio_type = 'cross_devices'
    else:
        scenrio_type = 'cross_silo'
    
    norm_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    resize = 256
    centercrop_size = 224
    preprocess = build_data_transform(norm_stats, resize, centercrop_size)
    
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='./Dataset', train=True, download=True, transform=preprocess
        )
        testset = torchvision.datasets.CIFAR10(
            root='./Dataset', train=False, download=True, transform=preprocess
        )
        num_classes = 10
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root='./Dataset', train=True, download=True, transform=preprocess
        )
        testset = torchvision.datasets.CIFAR100(
            root='./Dataset', train=False, download=True, transform=preprocess
        )
        num_classes = 100
    else:
        raise ValueError("Input dataset is not supported")
    # setup global testing set & distributed dataset
    testloader = Data.DataLoader(testset, batch_size=100, num_workers=2, shuffle=False)
    data_partitioner = DataPartitioner(trainset, args.n_clients)
        
    if args.alpha == 0.1:
        seed = 871
    elif args.alpha == 0.2:
        seed = 459
    elif args.alpha == 0.3:
        seed = 429
    elif args.alpha == 0.4:
        seed = 3760
    elif args.alpha == 0.5:
        seed = 448
    
    if args.data_distribution == 'non_iid_dirichlet':
        data_partitioner.dirichlet_split_noniid(alpha=args.alpha, least_samples=32, manual_seed=seed)
        #setup log file for recording
        npy_save_path = f"./output_record/loss_acc/info_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_alpha{args.alpha}"
        log_file_local_training = open(f"./output_record/log_output/local_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_alpha{args.alpha}.txt", mode="w+", encoding="utf-8")
        log_file_global_aggregation = open(f"./output_record/log_output/global_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}_alpha{args.alpha}.txt", 
                                           mode="a+", encoding="utf-8")
    elif args.data_distribution == 'manual_extreme_heterogeneity':
        data_partitioner.manual_allocating_noniid(args.n_dominated_class, 0.99, 1.0)
        #setup log file for recording
        npy_save_path = f"./output_record/loss_acc/info_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}"
        log_file_local_training = open(f"./output_record/log_output/local_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}.txt", mode="w+", encoding="utf-8")
        log_file_global_aggregation = open(f"./output_record/log_output/global_{args.alg}_{args.dataset}_{args.n_clients}clietns_{scenrio_type}_{args.data_distribution}.txt", 
                                           mode="a+", encoding="utf-8")
    else:
        raise ValueError("Input data distribution is not supported")
    #Get current time
    LATz = pytz.timezone("America/Los_Angeles") 
    timeInLA = datetime.now(LATz)
    print("-----------------------------------------------------------------------------------------", file=log_file_global_aggregation)
    print(f"*****Current time is {timeInLA} ***** Total communication rounds are {args.comms} *****", file=log_file_global_aggregation)
    print("-----------------------------------------------------------------------------------------", file=log_file_global_aggregation)
    #prepare to record eval loss and accuracy
    eval_loss_record = list()
    eval_acc_record = list()
    # call FL algorithm and run
    if args.alg in ['fedavg', 'fedprox', 'fedopt', 'scaffold', 'fedprox_gmm']:
        print(f"*****Current Federated Learning Scenario is {scenrio_type}*****", file=log_file_global_aggregation)
        # setup federated scenario
        fl_scen = FL_scenario(data_partitioner.get_all_client_weights(),
                              args.n_clients, args.n_sampled_clients,
                              data_partitioner.get_distributed_data(batch_size=args.batch))
        # construct model
        weight_init = 'random'
        server_model = to_device(Prompted_ViT_B32(weight_init=weight_init, 
                                                  prompt_method=args.prompt_method, 
                                                  num_tokens=args.n_tokens, 
                                                  num_classes=num_classes), args.device)
        server_model.build_trainable_keys()
        if args.alg == 'scaffold':
            server_model.init_contorl_parameter_for_scaffold(device=args.device)
        #get FL algorithm
        if '_' in args.alg:
            algclass = get_algorithm(args.alg.split('_')[0])
        else:
            algclass = get_algorithm(args.alg)
        if args.alg == 'fedprox':
            algo = algclass(server_model=server_model,scenario=fl_scen,
                        loss_fun=nn.CrossEntropyLoss(), mu=args.mu, device=args.device)
        elif args.alg == 'fedopt':
            algo = algclass(server_model=server_model,scenario=fl_scen,
                        loss_fun=nn.CrossEntropyLoss(), global_lr=args.fedopt_global_lr, device=args.device)
        elif args.alg == 'fedprox_gmm':
            print('gmm')
            algo = algclass(server_model=server_model,scenario=fl_scen,
                        loss_fun=nn.CrossEntropyLoss(), mu=args.mu, fed_method='simple_gmm_prompt', device=args.device)
        else:
            algo = algclass(server_model=server_model,scenario=fl_scen,
                            loss_fun=nn.CrossEntropyLoss(), device=args.device)
        
        for comm_round in range(args.comms):
            algo.client_train(comm_round=comm_round, epochs=args.local_eps, lr=args.lr, 
                              output_file=log_file_local_training, print_output=True)
            algo.server_aggre()
            print(f'--------------------------Round {comm_round+1} complete----------------------------',
                  file=log_file_local_training)
            eval_loss, eval_acc = algo.server_eval(testloader, comm_round, log_file_global_aggregation)
            eval_loss_record.append(eval_loss)
            eval_acc_record.append(eval_acc)
        min_index = np.argmin(np.array(eval_loss_record))
        print(f'*****Final accuracy with minimal eval_loss: {eval_acc_record[min_index]}*****', 
              file=log_file_global_aggregation)
        
    elif args.alg == 'pfedpg':
        trainloader = data_partitioner.get_distributed_data(batch_size=args.batch)
        # construct model
        clients = BaseHeadsForLocal(dataloaders=trainloader, num_classes=num_classes, local_lr=args.lr, device=args.device)
        prompt_gen = to_device(prompt_generator(num_tokens=args.n_tokens, 
                                                num_clients=args.n_clients, 
                                                k_dim=512, v_dim=512), args.device)
        vit_net = to_device(client_prompted_vit_b32(num_tokens=args.n_tokens), args.device)
        vit_net.build_trainable_keys()
        # setup distributed testset for personalized models
        test_partitioner = DataPartitioner(testset, args.n_clients)
        if args.data_distribution == 'non_iid_dirichlet':
            test_partitioner.dirichlet_split_noniid(args.alpha, least_samples=1, manual_seed=seed)
        elif args.data_distribution == 'manual_extreme_heterogeneity':
            test_partitioner.manual_allocating_noniid(args.n_dominated_class, 0.99, 1.0)
        else:
            raise ValueError("Input data distribution is not supported")
        distributed_testloaders = test_partitioner.get_distributed_data(batch_size=args.batch)
        eval_loss_record, eval_acc_record = train_eval_pFedPG(clients=clients,
                                                              prompt_gen=prompt_gen,
                                                              vit_net=vit_net,
                                                              comm_rounds=args.comms,
                                                              local_epochs=args.local_eps,
                                                              test_loaders=distributed_testloaders,
                                                              output_file=log_file_local_training,
                                                              inner_lr=args.lr,
                                                              prompt_lr=args.prompt_lr,
                                                              print_output=True,
                                                              device=args.device)
        evaluate_all_pFedPG(clients=clients, all_test_loader=testloader, prompt_gen=prompt_gen, vit_net=vit_net, 
                            output_file=log_file_global_aggregation, device=args.device)
    else:
        raise ValueError("Algorithm is not supported")
    #save record to npy file
    save_eval_npy_file(eval_loss_record, eval_acc_record, npy_save_path+'.npy')
    log_file_local_training.close()
    log_file_global_aggregation.close()
    
    
    