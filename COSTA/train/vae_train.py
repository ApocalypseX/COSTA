import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter

def train(model, loader, buffer, loss, optimizer, epoch_num, tb_path, model_path, test_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(tb_path)
    model.to(device)
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for epoch in range(epoch_num):
        model.train()
        train_total_loss=[]
        train_recon_loss=[]
        train_kld_loss=[]
        for i, (action, state) in enumerate(loader):
            action=action.to(device)
            state=state.to(device)
            pred=model(action,state)
            action_=pred[0]
            mu=pred[1]
            logv=pred[2]
            loss_dic=loss(action,action_,mu,logv)
            total_loss=loss_dic["total_loss"]
            recon_loss=loss_dic["recon_loss"]
            kld_loss=loss_dic["kld_loss"]
            train_total_loss.append(total_loss.item())
            train_recon_loss.append(recon_loss.item())
            train_kld_loss.append(kld_loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        mean_recon_loss=sum(train_recon_loss)/len(train_recon_loss)
        test_data=buffer.random_batch(test_size)
        testset = TensorDataset(torch.tensor(test_data["actions"], dtype=torch.float32), torch.tensor(test_data["observations"], dtype=torch.float32))
        testloader = DataLoader(testset, batch_size=1, shuffle=False)
        eval_loss=evaluate(model, testloader, loss)
        mean_total_loss=sum(train_total_loss)/len(train_total_loss)
        mean_kld_loss=sum(train_kld_loss)/len(train_kld_loss)
        mean_eval_kld_loss=eval_loss[0]
        max_eval_kld_loss=eval_loss[1]
        std_eval_kld_loss=eval_loss[2]
        mean_random_kld=eval_loss[3]
        min_random_kld=eval_loss[4]
        std_random_kld=eval_loss[5]
        writer.add_scalar(tag="loss/train_recon", scalar_value=mean_recon_loss,
                          global_step=epoch)
        writer.add_scalar(tag="loss/train_kld", scalar_value=mean_kld_loss,
                          global_step=epoch)
        writer.add_scalar(tag="loss/train_total", scalar_value=mean_total_loss,
                          global_step=epoch)
        writer.add_scalar(tag="kld/mean_eval", scalar_value=mean_eval_kld_loss,
                          global_step=epoch)
        writer.add_scalar(tag="kld/max_eval", scalar_value=max_eval_kld_loss,
                          global_step=epoch)
        writer.add_scalar(tag="kld/std_eval", scalar_value=std_eval_kld_loss,
                          global_step=epoch)
        writer.add_scalar(tag="kld/mean_random", scalar_value=mean_random_kld,
                          global_step=epoch)
        writer.add_scalar(tag="kld/min_random", scalar_value=min_random_kld,
                          global_step=epoch)
        writer.add_scalar(tag="kld/std_random", scalar_value=std_random_kld,
                          global_step=epoch)
        print('Trainging: epoch:%d, mean_recon_loss:%.5f, mean_kl_loss:%.5f, mean_total_loss:%.5f, \
                mean_eval_kl:%.5f, max_eval_kl:%.5f, std_eval_kl:%.5f, \
                mean_random_kl:%.5f, min_random_kl:%.5f, std_random_kl:%.5f' % 
              (epoch+1,mean_recon_loss,mean_kld_loss,mean_total_loss,mean_eval_kld_loss,max_eval_kld_loss,std_eval_kld_loss,
               mean_random_kld,min_random_kld,std_random_kld))
        torch.save(model, os.path.join(model_path, "epoch{}.pt".format(epoch+1)))
    writer.close()
    torch.save(model,os.path.join(model_path[:-21], "vae.pt"))
    return

def evaluate(model, loader, loss):
    # we only care about KL divergence here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    eval_kld_loss=[]
    random_kld_loss=[]
    for i, (action, state) in enumerate(loader):
        action=action.to(device)
        random_a=torch.zeros_like(action).uniform_(-1,1).to(device)
        state=state.to(device)
        pred=model(action,state)
        pred_r=model(random_a,state)
        action_=pred[0]
        mu=pred[1]
        logv=pred[2]
        
        r_action_=pred_r[0]
        r_mu=pred_r[1]
        r_logv=pred_r[2]
        
        loss_dic=loss(action,action_,mu,logv)
        loss_dic_r=loss(random_a,r_action_,r_mu,r_logv)
        kld_loss=loss_dic["kld_loss"]
        r_kld_loss=loss_dic_r["kld_loss"]
        eval_kld_loss.append(kld_loss.item())
        random_kld_loss.append(r_kld_loss.item())
    mean_kld_loss=sum(eval_kld_loss)/len(eval_kld_loss)
    max_kld_loss=max(eval_kld_loss)
    std_kld_loss=np.std(np.array(eval_kld_loss))
    
    mean_random_kld=sum(random_kld_loss)/len(random_kld_loss)
    min_random_kld=min(random_kld_loss)
    std_random_kld=np.std(np.array(random_kld_loss))
    return [mean_kld_loss,max_kld_loss,std_kld_loss,mean_random_kld,min_random_kld,std_random_kld]