import numpy as np
import os
import torch
import copy
import random
from torch import nn
from buffer import ReplayBuffer, SimpleSafeReplayBuffer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class CORROMetaLearner:
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
                 context_batch_size=10
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

        self.is_focal=is_focal
        self.context_batch_size=context_batch_size
        
        self.use_decoder=False
        self.use_cost_dynamic=False
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
        cost_context_first_sequence=[]
        for i in range(self.num_tasks):
            buffer=self.multi_task_buffer[i]
            if target=="train":
                j_num=self.meta_batch_size
                bs=self.context_batch_size
            elif target=="visual":
                j_num=self.visual_nums
                bs=self.ep_length
            for j in range(j_num):
                whole_sequence=buffer.random_batch(bs)
                cost_whole_context=torch.cat([whole_sequence["observations"], whole_sequence["actions"], 
                                        whole_sequence["next_observations"], whole_sequence["costs"]], dim=-1)
                cost_context.append(cost_whole_context)
        cost_context=torch.stack(cost_context, dim=0)
        return cost_context
    
    def distance_metric_loss(self, context):
        pos_loss=0.
        neg_loss=0.
        pos_cnt=0
        neg_cnt=0
        epsilon=1e-3
        bs,sl,dim=context.shape
        context=context.reshape(bs*sl,dim)
        model_out=self.encoder(context, mean=False)
        for i in range(self.meta_batch_size*self.num_tasks*self.context_batch_size):
            for j in range(i+1,self.meta_batch_size*self.num_tasks*self.context_batch_size):
                if (j//self.meta_batch_size*self.context_batch_size) == (i//self.meta_batch_size*self.context_batch_size):  #positive
                    pos_loss += torch.sqrt(torch.mean((model_out[i]-model_out[j])**2)+epsilon)
                    pos_cnt+=1
                else:
                    neg_loss += 1/(torch.sqrt(torch.mean((model_out[i]-model_out[j])**2)+epsilon/10000)+epsilon*100)
                    neg_cnt+=1
        return pos_loss/pos_cnt+neg_loss/neg_cnt
    
    #given the context, train one step
    def _take_step(self, context):
        context=context.to(self.device)
        dml_loss=self.distance_metric_loss(context)
        loss=dml_loss
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
        return loss_dict
                    
    # training offline RL, with evaluation on fixed eval tasks
    def train(self):
        self.encoder.to(self.device)
        for ep in range(self.num_epochs):
            self.encoder.train()
            dml_loss=[]
            lr=[]
            for it in range(self.train_steps_per_epoch):
                context=self.sample_context()
                loss_dict=self._take_step(context)
                dml_loss.append(loss_dict["dml_loss"])
                if "learning_rate" in loss_dict.keys():
                    lr.append(loss_dict["learning_rate"])
            if lr==[]:
                lr=[1e-4]
            train_stats={"dml_loss":sum(dml_loss)/len(dml_loss), "learning_rate":sum(lr)/len(lr)}
            self.log(ep+1,train_stats)
    
    def evaluate(self):
        pass

    def log(self, iteration, train_stats):
        #super().log(iteration, train_stats)
        if self.save_model and (iteration % self.save_interval == 0):
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
            torch.save(self.encoder, os.path.join(self.model_path, "encoder{0}.pt".format(iteration)))

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
        context=self.sample_context(target="visual")
        
        context=context.to(self.device)
        encodings=self.encoder(context)
        
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