import numpy as np
import os
import copy
import torch
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter

def train(model, loader, testloader, buffer, loss, optimizer, epoch_num, tb_path, model_path, test_size, root_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if root_path==None:
        root_path=model_path
    writer = SummaryWriter(tb_path)
    model.to(device)
    best_model=None
    best_eval_acc_unsafe=0
    best_eval_acc=0
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for epoch in range(epoch_num):
        model.train()
        train_total_loss=[]
        for i, (action, state, next_state, cost) in enumerate(loader):
            action=action.to(device)
            state=state.to(device)
            next_state=next_state.to(device)
            cost=cost.to(device)
            pred=model(torch.cat([state,action,next_state], -1))
            new_cost=cost.unsqueeze(0).repeat(pred.shape[0],1,1)

            total_loss=loss(pred,new_cost).mean(dim=(1,2)).sum()+model.get_decay_loss()
            train_total_loss.append(total_loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        eval_loss=evaluate(model, testloader)
        mean_total_loss=sum(train_total_loss)/len(train_total_loss)
        mean_eval_acc=eval_loss[0]
        mean_eval_acc_safe=eval_loss[1]
        mean_eval_acc_unsafe=eval_loss[2]
        if mean_eval_acc_unsafe>best_eval_acc_unsafe:
            best_eval_acc_unsafe=mean_eval_acc_unsafe
            best_eval_acc=mean_eval_acc
            best_model=copy.deepcopy(model)
        elif mean_eval_acc_unsafe>=best_eval_acc_unsafe-0.001 and mean_eval_acc>best_eval_acc:
            best_eval_acc_unsafe=mean_eval_acc_unsafe
            best_eval_acc=mean_eval_acc
            best_model=copy.deepcopy(model)
        writer.add_scalar(tag="loss/train_total", scalar_value=mean_total_loss,
                          global_step=epoch)
        writer.add_scalar(tag="kld/mean_eval", scalar_value=mean_eval_acc,
                          global_step=epoch)
        writer.add_scalar(tag="kld/mean_eval_safe", scalar_value=mean_eval_acc_safe,
                          global_step=epoch)
        writer.add_scalar(tag="kld/mean_eval_unsafe", scalar_value=mean_eval_acc_unsafe,
                          global_step=epoch)
        print('Trainging: epoch:%d, mean_total_loss:%.5f, \
                mean_eval_acc:%.5f, mean_eval_acc_safe:%.5f, mean_eval_acc_unsafe:%.5f' % 
              (epoch+1,mean_total_loss,mean_eval_acc,mean_eval_acc_safe,mean_eval_acc_unsafe))
        torch.save(model, os.path.join(model_path, "epoch{}.pt".format(epoch+1)))
    torch.save(best_model, os.path.join(root_path, "best_model.pt"))
    writer.close()
    return

def evaluate(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total_acc=0
    safe_acc=0
    unsafe_acc=0
    safe_count=0
    unsafe_count=0
    for i, (action, state, next_state, cost) in enumerate(loader):
        action=action.to(device)
        state=state.to(device)
        next_state=next_state.to(device)
        cost=cost[0]
        pred=model(torch.cat([state,action,next_state], -1)).mean(0)[0].item()
        if cost==1:
            unsafe_count+=1
            if pred>=0.5:
                total_acc+=1
                unsafe_acc+=1
        elif cost==0:
            safe_count+=1
            if pred<0.5:
                total_acc+=1
                safe_acc+=1
    print(total_acc,safe_acc,unsafe_acc)
    total_acc=total_acc/(safe_count+unsafe_count)
    safe_acc=safe_acc/safe_count
    unsafe_acc=unsafe_acc/unsafe_count
    return [total_acc,safe_acc,unsafe_acc]