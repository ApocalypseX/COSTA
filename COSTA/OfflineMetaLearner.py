import numpy as np
import os
import torch
import copy
import random
from torch import nn
from buffer import ReplayBuffer, SimpleSafeReplayBuffer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class OfflineMetaLearner:
    def __init__(self,
                 encoder,
                 decoder,
                 optimizer,
                 cost_dynamic_models,
                 num_tasks,
                 task_name,
                 observation_dim,
                 action_dim,
                 tb_writer,
                 model_path,
                 fig_path,
                 max_buffer_size=2000000,
                 meta_batch_size=8,
                 context_sequence_num=4,
                 goal_radius=3,
                 num_epochs=100,
                 visual_nums=50,
                 train_steps_per_epoch=100,
                 reward_std=0.1,
                 negative_num=10,
                 ep_length=300,
                 save_interval=10,
                 log_interval=1,
                 log_vis_interval=10,
                 save_model=True,
                 contrastive_weight=0.5,
                 decoder_weight=0.5,
                 scheduler=None,
                 is_focal=False,
                 use_dml_loss=True,
                 is_dis_constrained=True
                ):
        self.encoder=encoder
        self.decoder=decoder
        self.optimizer=optimizer
        self.cost_dynamic_models=cost_dynamic_models
        
        self.num_tasks=num_tasks
        self.task_name=task_name
        self.max_buffer_size=max_buffer_size
        self.observation_dim=observation_dim
        self.action_dim=action_dim
        self.goal_radius=goal_radius
        self.meta_batch_size=meta_batch_size
        self.visual_nums=visual_nums
        self.context_sequence_num=context_sequence_num

        self.scheduler=scheduler
        
        self.reward_std=reward_std
        self.ep_length=ep_length
        self.negative_num=negative_num
        self.contrastive_weight=contrastive_weight
        self.decoder_weight=decoder_weight
        
        self.num_epochs=num_epochs
        self.train_steps_per_epoch=train_steps_per_epoch
        
        self.tb_writer=tb_writer
        self.model_path=model_path
        self.fig_path=fig_path
        
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_model=save_model
        self.save_interval=save_interval
        self.log_interval=log_interval
        self.log_vis_interval=log_vis_interval

        self.use_dml_loss=use_dml_loss

        self.is_focal=is_focal
        
        self.use_decoder=False
        self.use_cost_dynamic=False

        self.is_dis_constrained = is_dis_constrained
        if self.decoder!=None:
            self.use_decoder=True
        
        if self.cost_dynamic_models!=None:
            self.use_cost_dynamic=True
        
        self.multi_task_buffer={}
        self.load_buffer()

    def load_buffer(self):
        for goal in range(self.num_tasks):
            data_path="offline_data/"+self.task_name+"_"+str(goal)+"/offline_buffer.npz"
            temp_buffer=SimpleSafeReplayBuffer(self.max_buffer_size, self.observation_dim, self.action_dim, self.goal_radius)
            temp_buffer.init_buffer(data_path)
            self.multi_task_buffer[goal]=temp_buffer
    
    def sample_context(self, target="train"):
        cost_context=[]
        reward_context=[]
        cost_context_first_sequence=[]
        reward_context_first_sequence=[]
        for i in range(self.num_tasks):
            buffer=self.multi_task_buffer[i]
            if target=="train":
                j_num=self.meta_batch_size
            elif target=="visual":
                j_num=self.visual_nums
            for j in range(j_num):
                if self.is_focal:
                    whole_sequence=buffer.random_batch(self.ep_length)
                    cost_whole_context=torch.cat([whole_sequence["observations"], whole_sequence["actions"], 
                                            whole_sequence["next_observations"], whole_sequence["costs"]], dim=-1)
                    cost_context.append(cost_whole_context)
                else:
                    whole_sequence, first_sequence=buffer.random_sequence(self.context_sequence_num, require_first_sequence=True)
                    cost_whole_context=torch.cat([whole_sequence["observations"], whole_sequence["actions"], 
                                            whole_sequence["next_observations"], whole_sequence["costs"]], dim=-1)
                    cost_first_context=torch.cat([first_sequence["observations"], first_sequence["actions"], 
                                            first_sequence["next_observations"], first_sequence["costs"]], dim=-1)
                    cost_context.append(cost_whole_context)
                    cost_context_first_sequence.append(cost_first_context)
                    reward_whole_context=torch.cat([whole_sequence["observations"], whole_sequence["actions"], 
                                            whole_sequence["next_observations"], whole_sequence["rewards"]], dim=-1)
                    reward_first_context=torch.cat([first_sequence["observations"], first_sequence["actions"], 
                                            first_sequence["next_observations"], first_sequence["rewards"]], dim=-1)
                    reward_context.append(reward_whole_context)
                    reward_context_first_sequence.append(reward_first_context)
        cost_context=torch.stack(cost_context, dim=0)
        if self.is_focal:
            return cost_context, None, None, None
        cost_context_first_sequence=torch.stack(cost_context_first_sequence, dim=0)
        reward_context=torch.stack(reward_context, dim=0)
        reward_context_first_sequence=torch.stack(reward_context_first_sequence, dim=0)
        return cost_context, cost_context_first_sequence, reward_context, reward_context_first_sequence
    
    def distance_metric_loss(self, context, context_first_sequence):
        pos_loss=0.
        neg_loss=0.
        pos_cnt=0
        neg_cnt=0
        epsilon=1e-3
        model_out=self.encoder(context,context_first_sequence)
        for i in range(self.meta_batch_size*self.num_tasks):
            for j in range(i+1,self.meta_batch_size*self.num_tasks):
                if (j//self.meta_batch_size) == (i//self.meta_batch_size):  #positive
                    pos_loss += torch.sqrt(torch.mean((model_out[i]-model_out[j])**2)+epsilon)
                    pos_cnt+=1
                else:
                    neg_loss += 1/(torch.sqrt(torch.mean((model_out[i]-model_out[j])**2)+epsilon/10000)+epsilon*100)
                    neg_cnt+=1
        if self.use_cost_dynamic:
            cpos_loss=0.
            cneg_loss=0.
            cpos_cnt=0
            cneg_cnt=0
            with torch.no_grad():
                n_context, n_context_first_sequence=self.create_negative_cost(context, context_first_sequence)
                if not self.is_dis_constrained:
                    new_context,new_context_first_sequence,_,_=self.sample_context()
                    n_context, n_context_first_sequence=self.create_negative_cost(new_context, new_context_first_sequence)
                p_context, p_context_first_sequence=self.create_positive_cost()
            bs=n_context.shape[0]
            sl=n_context.shape[2]
            p_model_out=self.encoder(p_context, p_context_first_sequence)
            contrastive_model_out=self.encoder(n_context.reshape(bs*self.negative_num,sl,-1), n_context_first_sequence.reshape(bs*self.negative_num,sl,-1)).reshape(bs,self.negative_num,-1)
            for i in range(model_out.shape[0]):
                for j in range(self.meta_batch_size*self.num_tasks):
                    if (j//self.meta_batch_size) == (i//self.meta_batch_size):
                        cpos_loss += torch.sqrt(torch.mean((model_out[i]-p_model_out[j])**2)+epsilon)
                        cpos_cnt+=1
                        cpos_loss += torch.sqrt(torch.mean((p_model_out[i]-p_model_out[j])**2)+epsilon)
                        cpos_cnt+=1
                        for k in range(contrastive_model_out.shape[1]):
                            cneg_loss += 1/(torch.sqrt(torch.mean((model_out[i]-contrastive_model_out[j][k])**2)+epsilon/10000)+epsilon*100)
                            cneg_cnt+=1
                    else:
                        cneg_loss += 1/torch.sqrt(torch.mean((p_model_out[i]-p_model_out[j])**2)+epsilon/100)
                        cneg_cnt+=1

            dml_loss=pos_loss/pos_cnt+neg_loss/neg_cnt
            contrastive_loss=cpos_loss/cpos_cnt+cneg_loss/cneg_cnt
            total_loss=(pos_loss+self.contrastive_weight*cpos_loss)/(pos_cnt+self.contrastive_weight*cpos_cnt)+(neg_loss+self.contrastive_weight*cneg_loss)/(neg_cnt+self.contrastive_weight*cneg_cnt)
            if dml_loss.item()==np.NaN or contrastive_loss.item()==np.NaN:
                print(pos_loss,neg_loss,cpos_loss,cneg_loss)
            return dml_loss, contrastive_loss, total_loss, model_out
            
        return pos_loss/pos_cnt+neg_loss/neg_cnt, None, pos_loss/pos_cnt+neg_loss/neg_cnt, model_out
    
    def sample_positive_batch(self, task, num=1):
        buffer=self.multi_task_buffer[task]
        cost_context=[]
        reward_context=[]
        cost_context_first_sequence=[]
        reward_context_first_sequence=[]
        for j in range(num):
            if self.is_focal:
                whole_sequence=buffer.random_batch(self.ep_length)
                cost_whole_context=torch.cat([whole_sequence["observations"], whole_sequence["actions"], 
                                        whole_sequence["next_observations"], whole_sequence["costs"]], dim=-1)
                cost_context.append(cost_whole_context)
            else:
                whole_sequence, first_sequence=buffer.random_sequence(self.context_sequence_num, require_first_sequence=True)
                cost_whole_context=torch.cat([whole_sequence["observations"], whole_sequence["actions"], 
                                        whole_sequence["next_observations"], whole_sequence["costs"]], dim=-1)
                cost_first_context=torch.cat([first_sequence["observations"], first_sequence["actions"], 
                                        first_sequence["next_observations"], first_sequence["costs"]], dim=-1)
                cost_context.append(cost_whole_context)
                cost_context_first_sequence.append(cost_first_context)
                reward_whole_context=torch.cat([whole_sequence["observations"], whole_sequence["actions"], 
                                        whole_sequence["next_observations"], whole_sequence["rewards"]], dim=-1)
                reward_first_context=torch.cat([first_sequence["observations"], first_sequence["actions"], 
                                        first_sequence["next_observations"], first_sequence["rewards"]], dim=-1)
                reward_context.append(reward_whole_context)
                reward_context_first_sequence.append(reward_first_context)
        cost_context=torch.stack(cost_context, dim=0)
        if self.is_focal:
            return cost_context, None, None, None
        cost_context_first_sequence=torch.stack(cost_context_first_sequence, dim=0)
        reward_context=torch.stack(reward_context, dim=0)
        reward_context_first_sequence=torch.stack(reward_context_first_sequence, dim=0)
        return cost_context, cost_context_first_sequence, reward_context, reward_context_first_sequence
    
    #simply use reward randomization as negative reward
    def create_negative_reward(self, context, context_first_sequence):
        context=context.unsqueeze(1).expand(-1,self.negative_num,-1,-1)
        context_first_sequence=context_first_sequence.unsqueeze(1).expand(-1,self.negative_num,-1,-1)
        org_context_reward=context[...,-1:].clone()
        reward_noise=torch.normal(0, self.reward_std, org_context_reward.shape)
        context[...,-1:]=org_context_reward+reward_noise
        context_first_sequence[...,-1:]=context_first_sequence[...,-1:]+reward_noise[:,:,:self.ep_length,:]
        return context, context_first_sequence
    
    #using other task's discriminator model to create negative cost 
    def create_negative_cost(self, context, context_first_sequence):
        context=copy.deepcopy(context)
        context_first_sequence=copy.deepcopy(context_first_sequence)
        context=context.unsqueeze(1).expand(-1,self.negative_num,-1,-1)
        #context_first_sequence=context_first_sequence.unsqueeze(1).expand(-1,self.negative_num,-1,-1)
        for i in range(self.meta_batch_size*self.num_tasks):
            task=(i//self.meta_batch_size)
            choices=list(range(self.num_tasks))
            choices.remove(task)
            for j in range(self.negative_num):
                #for k in range(int(context.shape[-2]/10)):
                discriminator=self.cost_dynamic_models[random.choice(choices)]
                discriminator_input=context[i,j,:,:-1]
                new_cost=discriminator(discriminator_input).mean(0)
                context[i,j,:,-1:]=torch.where(new_cost>=0.5,torch.tensor(1.0).to(self.device),torch.tensor(0.0).to(self.device))
        context_first_sequence=copy.deepcopy(context[:,:,:self.ep_length,:])
        return context, context_first_sequence
    
    def create_positive_cost(self):
        cost_context=[]
        cost_context_first_sequence=[]
        for i in range(self.num_tasks):
            tasks=list(range(self.num_tasks))
            dis_model=self.cost_dynamic_models[i]
            tasks.remove(i)
            for j in range(self.meta_batch_size):
                index=random.choice(tasks)
                buffer=self.multi_task_buffer[index]
                if self.is_focal:
                    whole_sequence=buffer.random_batch(self.ep_length)
                else:
                    whole_sequence, first_sequence=buffer.random_sequence(self.context_sequence_num, require_first_sequence=True)
                cost_whole_context=torch.cat([whole_sequence["observations"], whole_sequence["actions"], 
                                         whole_sequence["next_observations"], whole_sequence["costs"]], dim=-1)
                discriminator_input=cost_whole_context[...,:-1]
                new_cost=dis_model(discriminator_input).mean(0)
                cost_whole_context[...,-1:]=torch.where(new_cost>=0.5,torch.tensor(1.0).to(self.device),torch.tensor(0.0).to(self.device))
                cost_context.append(cost_whole_context)
        cost_context=torch.stack(cost_context, dim=0)
        cost_context_first_sequence=copy.deepcopy(cost_context[:,:self.ep_length,:])
        return cost_context, cost_context_first_sequence
    
    #given the context, train one step
    def _take_step(self, context, context_first_sequence):
        context=context.to(self.device)
        if context_first_sequence is not None:
            context_first_sequence=context_first_sequence.to(self.device)
        dml_loss, con_loss, total_loss, model_out=self.distance_metric_loss(context,context_first_sequence)
        #loss=total_loss
        if  not self.use_cost_dynamic:
            loss=dml_loss
        else:
            loss=dml_loss+self.contrastive_weight*con_loss
        if not self.use_dml_loss:
            loss=self.contrastive_weight*con_loss
        assert loss is not np.NaN
        if self.use_decoder:
            loss_f=nn.BCELoss()
            model_out=model_out.unsqueeze(1).expand(-1,self.ep_length*self.context_sequence_num,-1)
            decoder_input=torch.cat([context[...,:-1],model_out],dim=-1)
            decoder_label=context[...,-1:]
            decoder_output=self.decoder(decoder_input)
            decoder_loss=loss_f(decoder_output,decoder_label)
            loss+=decoder_loss*self.decoder_weight
        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(self.encoder.parameters(), max_norm=5, norm_type=2)
        # if self.use_decoder:
        #     nn.utils.clip_grad_norm(self.decoder.parameters(), max_norm=5, norm_type=2)
        self.optimizer.step()
        if self.scheduler is not None:
            if self.scheduler.get_lr()[0]>=3e-5:
                self.scheduler.step()
            loss_dict={"dml_loss":dml_loss.item(), "learning_rate":self.scheduler.get_lr()[0]}
        else:
            loss_dict={"dml_loss":dml_loss.item()}
        if self.use_decoder:
            loss_dict["cost_pred_loss"]=decoder_loss.item()
        if self.use_cost_dynamic:
            loss_dict["contrastive_loss"]=con_loss.item()
            loss_dict["total_dml_loss"]=total_loss.item()
        return loss_dict
                    
    # training offline RL, with evaluation on fixed eval tasks
    def train(self):
        self.encoder.to(self.device)
        if self.use_decoder:
            self.decoder.to(self.device)
        for ep in range(self.num_epochs):
            self.encoder.train()
            if self.use_decoder:
                self.decoder.train()
            dml_loss=[]
            total_dml_loss=[]
            lr=[]
            if self.use_decoder:
                decoder_loss=[]
            if self.use_cost_dynamic:
                contrastive_loss=[]
            for it in range(self.train_steps_per_epoch):
                context,context_first_sequence,_,_=self.sample_context()
                loss_dict=self._take_step(context,context_first_sequence)
                dml_loss.append(loss_dict["dml_loss"])
                if "total_dml_loss" in loss_dict.keys():
                    total_dml_loss.append(loss_dict["total_dml_loss"])
                if "learning_rate" in loss_dict.keys():
                    lr.append(loss_dict["learning_rate"])
                if self.use_decoder:
                    decoder_loss.append(loss_dict["cost_pred_loss"])
                if self.use_cost_dynamic:
                    contrastive_loss.append(loss_dict["contrastive_loss"])
            if total_dml_loss==[]:
                total_dml_loss=dml_loss
            if lr==[]:
                lr=[1e-4]
            train_stats={"dml_loss":sum(dml_loss)/len(dml_loss), "total_dml_loss":sum(total_dml_loss)/len(total_dml_loss), "learning_rate":sum(lr)/len(lr)}
            if self.use_decoder:
                train_stats["cost_pred_loss"]=sum(decoder_loss)/len(decoder_loss)
            if self.use_cost_dynamic:
                train_stats["contrastive_loss"]=sum(contrastive_loss)/len(contrastive_loss)
            self.log(ep+1,train_stats)
    
    def evaluate(self):
        pass

    def log(self, iteration, train_stats):
        #super().log(iteration, train_stats)
        if self.save_model and (iteration % self.save_interval == 0):
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
            torch.save(self.encoder, os.path.join(self.model_path, "encoder{0}.pt".format(iteration)))
            if self.use_decoder:
                torch.save(self.decoder, os.path.join(self.model_path, "decoder{0}.pt".format(iteration)))

        if iteration % self.log_interval == 0:
            for k in train_stats.keys():
                self.tb_writer.add_scalar('encoder_losses/'+k, train_stats[k], 
                    self.train_steps_per_epoch*iteration)
            print("Iteration -- {}"
                .format(iteration), train_stats)

        # visualize embeddings
        if iteration % self.log_vis_interval == 0:
            if not os.path.exists(self.fig_path):
                os.mkdir(self.fig_path)
            self.vis_sample_embeddings(os.path.join(self.fig_path, "train_fig{0}.png".format(iteration)))
    
    def vis_sample_embeddings(self, save_path):
        self.encoder.eval()
        x, y = [], []
        context,context_first_sequence,_,_=self.sample_context(target="visual")
        
        context=context.to(self.device)
        if context_first_sequence is not None:
            context_first_sequence=context_first_sequence.to(self.device)
        encodings=self.encoder(context,context_first_sequence)
        
        encodings = encodings.cpu().detach().numpy()

        for i in range(self.visual_nums*self.num_tasks):
            x.append(encodings[i])
            y.append(i//self.visual_nums)
        
        
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(np.asarray(x))

        x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
        data = (X_tsne - x_min) / (x_max - x_min)

        colors = plt.cm.rainbow(np.linspace(0,1,self.num_tasks))
        #print(colors)
        
        plt.cla()
        fig = plt.figure()
        ax = plt.subplot(111)
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(y[i]),
                    color=colors[y[i]], #plt.cm.Set1(y[i] / 21),
                    fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_path)