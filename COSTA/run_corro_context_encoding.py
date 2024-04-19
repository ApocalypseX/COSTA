import numpy as np
import glob
import os
import abc
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import torch
import json
import random
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter
from sacred import Experiment
from buffer import ReplayBuffer, SimpleSafeReplayBuffer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from nets import *
from CORROMetaLearner import CORROMetaLearner
from utils.common import seed_torch,read_json_dict

ex = Experiment()

@ex.capture(prefix='context_params')
def run(_config):
    seed_torch(_config["seed"])
    observation_dim=_config["state_dim"]
    action_dim=_config["action_dim"]
    encode_dim=_config["encode_dim"]
    #encoder=SelfAttnEncoder(2*observation_dim+action_dim+1,_config["encoder_hidden_dim"],_config["encoder_hidden_layers"],output_dim=encode_dim)
    if _config["pretrained_encoder"]!=None:
        encoder=torch.load(_config["pretrained_encoder"])
    #encoder=RNNEncoder(2*observation_dim+action_dim+1,_config["encoder_hidden_dim"],_config["encoder_hidden_layers"],output_dim=encode_dim)
    #encoder=MLPAttnEncoder(input_dim=2*observation_dim+action_dim+1,hidden_dims=_config["hidden_dims"],output_dim=encode_dim)
    encoder=MLPEncoder(input_dim=2*observation_dim+action_dim+1,hidden_dims=_config["hidden_dims"],output_dim=encode_dim)
    #decoder=None
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=_config["learning_rate"], eps=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=_config["decay_step"], gamma=_config["decay_rate"])
    #scheduler=None
    num_tasks=_config["num_tasks"]
    tb_path=_config["path"]+"/tb"
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    tb_writer=SummaryWriter(tb_path)
    model_path=_config["path"]+"/model"
    fig_path=_config["path"]+"/fig"
    task_name=_config["env_name"]
    learner=CORROMetaLearner(encoder,
                           None,
                           optimizer,
                           None,
                           num_tasks,
                           task_name,
                           observation_dim,
                           action_dim,
                           tb_writer,
                           model_path,
                           fig_path,
                           context_sequence_num=_config["context_sequence_num"],
                           max_buffer_size=_config["max_buffer_size"],
                           meta_batch_size=_config["meta_batch_size"],
                           num_epochs=_config["num_epochs"],
                           train_steps_per_epoch=_config["train_steps_per_epoch"],
                           reward_std=_config["reward_std"],
                           negative_num=_config["negative_num"],
                           ep_length=_config["ep_length"],
                           save_interval=_config["save_interval"],
                           log_interval=_config["log_interval"],
                           log_vis_interval=_config["log_vis_interval"],
                           contrastive_weight=_config["contrastive_weight"],
                           decoder_weight=_config["decoder_weight"],
                           scheduler=scheduler,
                           is_focal=False,
                           context_batch_size=4
                          )
    learner.train()

@ex.main
def my_main():
    run()

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='ant-dir-safe')
    args = parser.parse_args()

    config_dict=read_json_dict("configs/"+args.task+".json")
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
    config_dict["context_params"]["path"]=os.path.join(config_dict["context_params"]["path"],args.task,"corro_context_encoder",mkfile_time)
    root_path=config_dict["context_params"]["path"]
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    ex.add_config(config_dict)
    file_obs_path = os.path.join(root_path,"sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run()