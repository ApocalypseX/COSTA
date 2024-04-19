import argparse
import random
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from torch.backends import cudnn
import gym
import numpy as np
import torch
from nets import MLP
from modules import ActorProb, Critic, TanhDiagGaussian
from buffer import SimpleSafeReplayBuffer
from utils.logger import Logger, make_log_dirs
from policy_trainer import MFPEARLSafePolicyTrainer
from policy import PEARLCPQPolicy
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs import ENVS
from model.vae_model import *
from lagrange.meta_lagrange import MetaLagrange
from lagrange.lagrange import Lagrange
from lagrange.pid_lagrange import PIDLagrangian
from utils.common import seed_torch, read_json_dict
from nets.encoder import *

ex = Experiment()

@ex.capture
def train(_config):
    seed=_config["seed"]
    seed_torch(seed)
    # create env and dataset
    num_tasks=_config["num_tasks"]
    env_list=[]
    meta_buffer={}
    vae_list = []
    for task in range(num_tasks):
        env = NormalizedBoxEnv(ENVS[_config['env_name']](**_config['env_params']))
        env.seed(seed)
        env.reset_task(task)
        env_list.append(env)
        temp_buffer=SimpleSafeReplayBuffer(_config["max_buffer_size"], _config["state_dim"], _config["action_dim"], _config["goal_radius"])
        data_path="offline_data/"+_config['env_name']+"_"+str(task)+"/offline_buffer.npz"
        temp_buffer.init_buffer(data_path, cost_bound=_config["safety_threshold"])
        meta_buffer[task]=temp_buffer
        temp_vae=torch.load("run/"+_config['env_name']+"_"+str(task)+"/vae/vae.pt")
        vae_list.append(temp_vae)

    state_dim=_config["state_dim"]+_config["context_params"]["encode_dim"]
    observation_dim=_config["state_dim"]
    action_dim=_config["action_dim"]
    if _config["use_state_augmentation"]:
        state_dim+=1
    # create policy model
    actor_backbone = MLP(input_dim=np.prod(state_dim), hidden_dims=_config["hidden_dim"])
    critic1_backbone = MLP(input_dim=np.prod(state_dim) + _config["action_dim"], hidden_dims=_config["hidden_dim"])
    critic2_backbone = MLP(input_dim=np.prod(state_dim) + _config["action_dim"], hidden_dims=_config["hidden_dim"])
    critic_c_backbone = MLP(input_dim=np.prod(state_dim) + _config["action_dim"], hidden_dims=_config["hidden_dim"])
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=_config["action_dim"],
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, _config["device"])
    critic1 = Critic(critic1_backbone, _config["device"])
    critic2 = Critic(critic2_backbone, _config["device"])
    critic_c = Critic(critic_c_backbone, _config["device"])
    actor_optim = torch.optim.Adam(actor.parameters(), lr=_config["actor_lr"])
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=_config["critic_lr"])
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=_config["critic_lr"])
    critic_c_optim = torch.optim.Adam(critic_c.parameters(), lr=_config["critic_c_lr"])

    encoder = MLPUDEncoder(input_dim=2*observation_dim+action_dim+1, hidden_dims=_config["context_params"]["hidden_dims"], output_dim=_config["context_params"]["encode_dim"])
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=_config["critic_lr"])

    if _config["safety_lagrange_pid"]:
        lagrange = PIDLagrangian(**_config["pid_lagrange_params"])
    else:
        lagrange_ls=[]
        for goal in range(num_tasks):
            lagrange=Lagrange(**_config["lagrange_params"])
            lagrange_ls.append(lagrange)

    if _config["auto_alpha"]:
        target_entropy = _config["target_entropy"] if _config["target_entropy"] \
            else -np.prod(env.action_space.shape)

        log_alpha = torch.zeros(1, requires_grad=True, device=_config["device"])
        alpha_optim = torch.optim.Adam([log_alpha], lr=_config["alpha_lr"])
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = _config["alpha"]

    # create policy
    policy = PEARLCPQPolicy(
        actor,
        critic1,
        critic2,
        critic_c,
        vae_list,
        encoder,
        actor_optim,
        critic1_optim,
        critic2_optim,
        critic_c_optim,
        encoder_optim,
        action_space=env_list[0].action_space,
        lagrange=lagrange_ls,
        tau=_config["tau"],
        gamma=_config["gamma"],
        alpha=alpha,
        cql_weight=_config["cql_weight"],
        cpq_weight=_config["cpq_weight"],
        temperature=_config["temperature"],
        max_q_backup=_config["max_q_backup"],
        deterministic_backup=_config["deterministic_backup"],
        with_lagrange=_config["with_lagrange"],
        use_vae=_config["use_vae"],
        train_cpq_alpha=_config["train_cpq_alpha"],
        lagrange_threshold=_config["lagrange_threshold"],
        cql_alpha_lr=_config["cql_alpha_lr"],
        num_repeart_actions=_config["num_repeat_actions"],
        safety_threshold=_config["safety_threshold"],
        kl_threshold=_config["kl_threshold"],
        cpq_alpha_lr=_config["cpq_alpha_lr"],
        policy_train=_config["policy_train"],
        use_state_augmentation=_config["use_state_augmentation"],
        use_safety_lagrange=_config["use_safety_lagrange"],
        safety_lagrange_pid=_config["safety_lagrange_pid"],
        warm_up_epoch=_config["warm_up_epoch"],
        use_conservative_reward_loss=_config["use_conservative_reward_loss"],
        use_conservative_cost_loss=_config["use_conservative_cost_loss"],
        lgr_lower_bound=_config["lgr_lower_bound"],
        lgr_upper_bound=_config["lgr_upper_bound"]
    )

    # log
    #log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    log_dirs = _config["log_dirs"]
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(_config))
    # create policy trainer
    policy_trainer = MFPEARLSafePolicyTrainer(
        policy=policy,
        eval_env_list=env_list,
        num_tasks=num_tasks,
        buffer=meta_buffer,
        logger=logger,
        epoch=_config["epoch"],
        step_per_epoch=_config["step_per_epoch"],
        batch_size=_config["batch_size"],
        sequence_batch_size=_config["sequence_batch_size"],
        eval_episodes=_config["eval_episodes"],
        use_state_augmentation= _config["use_state_augmentation"],
        safety_bound=_config["safety_threshold"],
        use_sequence_batch=_config["use_sequence_batch"],
        context_sequence_num=_config["context_sequence_num"],
        is_focal=True
    )

    # train
    policy_trainer.train()

@ex.main
def my_main():
    train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='ant-dir-safe')
    parser.add_argument('--seed', type=int, default=-1)
    args = parser.parse_args()
    config_dict=read_json_dict("configs/"+args.task+".json")
    if args.seed>=0:
        config_dict["seed"]=args.seed
    log_dirs = make_log_dirs(args.task, "pearl_cpq", config_dict["seed"], config_dict)
    ex.add_config(config_dict)
    ex.add_config({"log_dirs":log_dirs})
    file_obs_path = os.path.join(log_dirs,"sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run()