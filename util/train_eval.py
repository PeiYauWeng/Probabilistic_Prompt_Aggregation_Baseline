import torch
import torch.optim as optim
import random
import copy
import numpy as np
from util.print_info import print_epoch_end

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def scaffold_tuning_gradients(model, server_controls, client_controls):
    for name, param in model.named_parameters():
        if param.requires_grad==True:
            param.grad.data += (server_controls[name] - client_controls[name])
    
def train(model, data_loader, optimizer, loss_fun, device='cuda'):
    model.train()
    loss_all = 0
    total = 0
    accuracy = 0
    current_lr = get_lr(optimizer)
    for data, target in data_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        accuracy += pred.eq(target.view(-1)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return current_lr, loss_all/len(data_loader), accuracy/total

def evaluate(model, data_loader, loss_fun, device='cuda'):
    model.eval()
    loss_all = 0
    total = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            output = model(data)
            loss = loss_fun(output, target)
            loss_all += loss.item()
            total += target.size(0)
            pred = output.data.max(1)[1]
            accuracy += pred.eq(target.view(-1)).sum().item()
    return loss_all / len(data_loader), accuracy/total

def evaluate_pFedPG(eval_loss_record, eval_acc_record, clients, selected_clients, test_loaders, 
                    prompt_gen, vit_net, criteria, output_file, print_out=False, device='cuda'):
    prompt_gen.eval()
    vit_net.eval()
    #accuracy_storage = list()
    for cl_id in selected_clients:
        running_loss, running_correct = 0.0, 0.0
        total = 0
        curr_test_loader = test_loaders[cl_id]
        gen_prompt = prompt_gen(x_id=torch.tensor([cl_id], dtype=torch.long).to(device))
        vit_net.prompt_embeddings.data.copy_(gen_prompt.data)
        clients.local_layers[cl_id].eval()
        for img, label in curr_test_loader:
            img = img.to(device).float()
            label = label.to(device).long()
            out = vit_net(img)
            out = clients.local_layers[cl_id](out)
            running_loss += criteria(out, label).item()
            pred = out.data.max(1)[1]
            running_correct += pred.eq(label.view(-1)).sum().item()
            total += label.size(0)
        running_loss = running_loss/len(curr_test_loader)
        running_correct = running_correct/total
        if print_out:
            print("Client {}, eval_loss: {:.4f}, accuracy: {:.2f}".format(cl_id+1, running_loss, running_correct),
                  file=output_file)
        eval_loss_record[cl_id].append(running_loss)
        eval_acc_record[cl_id].append(running_correct)
        #accuracy_storage.append(running_correct)
    #if len(selected_clients) == len(clients):
        #average_accuracy = np.mean(np.array(accuracy_storage))
        #print("Average Accuracy for distributed testset: {:.4f}".format(average_accuracy), file=output_file)
        
def evaluate_all_pFedPG(clients, all_test_loader, prompt_gen, vit_net, output_file,
                        criteria=torch.nn.CrossEntropyLoss(), device='cuda'):
    prompt_gen.eval()
    vit_net.eval()
    accuracy_storage = list()
    for cl_id in range(len(clients)):
        running_loss, running_correct = 0.0, 0.0
        total = 0
        gen_prompt = prompt_gen(x_id=torch.tensor([cl_id], dtype=torch.long).to(device))
        vit_net.prompt_embeddings.data.copy_(gen_prompt.data)
        clients.local_layers[cl_id].eval()
        for img, label in all_test_loader:
            img = img.to(device).float()
            label = label.to(device).long()
            out = vit_net(img)
            out = clients.local_layers[cl_id](out)
            running_loss += criteria(out, label).item()
            pred = out.data.max(1)[1]
            running_correct += pred.eq(label.view(-1)).sum().item()
            total += label.size(0)
        running_loss = running_loss/len(all_test_loader)
        running_correct = running_correct/total
        print("Client {}, eval_loss: {:.4f}, accuracy: {:.2f}".format(cl_id+1, running_loss, running_correct),
              file=output_file)
        accuracy_storage.append(running_correct)
    average_accuracy = np.mean(np.array(accuracy_storage))
    max_accuracy = np.max(np.array(accuracy_storage))
    print("Average Accuracy for global testset: {:.4f}".format(average_accuracy), file=output_file)
    print("Maximal Accuracy for global testset: {:.4f}".format(max_accuracy), file=output_file)

def train_prox(model, server_model, data_loader, optimizer, loss_fun, mu, device='cuda'):
    model.train()
    loss_all = 0
    total = 0
    accuracy = 0
    current_lr = get_lr(optimizer)
    for step, (data, target) in enumerate(data_loader):
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        if step > 0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(filter(lambda p : p.requires_grad, server_model.parameters()),
                              filter(lambda p : p.requires_grad, model.parameters())):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            w_diff = torch.sqrt(w_diff)
            loss += mu / 2. * w_diff
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        accuracy += pred.eq(target.view(-1)).sum().item()
    return current_lr, loss_all / len(data_loader), accuracy/total

def train_scaffold(model, server_model, data_loader, optimizer, loss_fun, local_epochs, device='cuda'):
    model.train()
    loss_all = 0
    total = 0
    accuracy = 0
    current_lr = get_lr(optimizer)
    for data, target in data_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        accuracy += pred.eq(target.view(-1)).sum().item()
        optimizer.zero_grad()
        loss.backward()
        #optimizer.step(server_model.control, model.control)
        scaffold_tuning_gradients(model, server_model.control, model.control)
        optimizer.step()
    return current_lr, loss_all / len(data_loader), accuracy/total

def train_eval_pFedPG(clients, prompt_gen, vit_net, comm_rounds, local_epochs, test_loaders, output_file,
                      inner_lr, prompt_lr, device='cuda', print_output=False):
    optimizer = optim.Adam(params=prompt_gen.parameters(), lr=prompt_lr)
    criteria = torch.nn.CrossEntropyLoss()
    eval_loss_record = [list() for _ in range(len(clients))]
    eval_acc_record = [list() for _ in range(len(clients))]
    
    #selected_clients = list()
    for step in range(comm_rounds):
        prompt_gen.train()
        #select client at random
        client_id = random.choice(range(len(clients)))
        #selected_clients.append(client_id)
        print(f'------------Client_{client_id+1} start trainig------------', file=output_file)
        #generate personalized prompt & load local vit_net
        generated_prompt = prompt_gen(x_id=torch.tensor([client_id], dtype=torch.long).to(device))
        vit_net.prompt_embeddings.data.copy_(generated_prompt.data)

        #init inner optimizer
        inner_optim = optim.Adam(filter(lambda p : p.requires_grad, vit_net.parameters()), lr=inner_lr,
                                 betas=(0.9, 0.98), eps=1e-6)
        # storing theta_i for later calculating delta theta
        inner_state = copy.deepcopy(vit_net.prompt_embeddings.data)

        for i in range(local_epochs):
            vit_net.train()
            clients.local_layers[client_id].train()
            loss_all = 0.0
            accuracy = 0.0
            total = 0
            current_lr = get_lr(clients.local_optimizers[client_id])
            for img, label in clients.dataloaders[client_id]:
                inner_optim.zero_grad()
                #optimizer.zero_grad()
                clients.local_optimizers[client_id].zero_grad()
                img = img.to(device).float()
                label = label.to(device).long()

                net_out = vit_net(img)
                output = clients.local_layers[client_id](net_out)

                loss = criteria(output, label)
                loss.backward()
                inner_optim.step()
                clients.local_optimizers[client_id].step()

                loss_all += loss.item()
                total += label.size(0)
                pred = output.data.max(1)[1]
                accuracy += pred.eq(label.view(-1)).sum().item()
            loss_all = loss_all/len(clients.dataloaders[client_id])
            accuracy = accuracy/total
            if print_output:
                print_epoch_end(i, current_lr, loss_all, accuracy, output_file)

        optimizer.zero_grad()
        final_state = copy.deepcopy(vit_net.prompt_embeddings.data)
        # calculating delta theta
        delta_theta = inner_state - final_state
        # calculating phi gradient
        prompt_gen_grads = torch.autograd.grad(
            list(generated_prompt), prompt_gen.parameters(), grad_outputs=list(delta_theta)
        )
        # update prompt_generator weights
        for p, g in zip(prompt_gen.parameters(), prompt_gen_grads):
            p.grad = g
        optimizer.step()

        if (step+1)%10 == 0 or (step+1)==comm_rounds:
            print_out=False
            print(f'--------------------------Comm_Round_{step+1} complete--------------------------', file=output_file)
            if (step+1)==comm_rounds:
                print_out=True
            evaluate_pFedPG(eval_loss_record=eval_loss_record,
                            eval_acc_record=eval_acc_record,
                            clients=clients,
                            selected_clients=list(range(len(clients))),
                            test_loaders=test_loaders,
                            prompt_gen=prompt_gen,
                            vit_net=vit_net,
                            criteria=criteria,
                            output_file=output_file,
                            print_out=print_out)
    # print evaluate all clients and get average accuracy
    accuracy_storage = list()
    for j in range(len(clients)):
        accuracy_storage.append(eval_acc_record[j][-1])
    average_accuracy = np.mean(np.array(accuracy_storage))
    print("Average Accuracy for global testset: {:.4f}".format(average_accuracy), file=output_file)
            
    return eval_loss_record, eval_acc_record