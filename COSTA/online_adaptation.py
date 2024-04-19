import argparse
import os
import copy
import numpy as np
import torch
from nets import MLP
from modules import ActorProb, Critic, TanhDiagGaussian
from utils.logger import Logger, make_log_dirs
from policy_trainer import MFMetaSafePolicyTrainer
from buffer import SimpleSafeReplayBuffer
from policy import MetaCPQPolicy
from sacred import Experiment
from sacred.observers import FileStorageObserver
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs import ENVS
from model.vae_model import *
from utils.common import seed_torch, read_json_dict
from nets.encoder import *
import matplotlib.pyplot as plt

ex = Experiment()

@ex.capture
def do_adaptation(_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed=_config["seed"]
    seed_torch(seed)
    # create env and dataset
    num_tasks=_config["num_tasks"]
    env_list=[]
    vae_list = []
    meta_buffer={}
    for task in range(num_tasks):
        env = NormalizedBoxEnv(ENVS[_config['env_name']](**_config['env_params']))
        env.seed(seed)
        if _config["ood"]:
            env.reset_task(task+num_tasks)
        else:
            env.reset_task(task)
        env_list.append(env)
        temp_vae=torch.load("run/"+_config['env_name']+"_"+str(task)+"/vae/vae.pt")
        vae_list.append(temp_vae)
        data_path="offline_data/"+_config['env_name']+"_"+str(task)+"/offline_buffer.npz"
        temp_buffer=SimpleSafeReplayBuffer(_config["max_buffer_size"], _config["state_dim"], _config["action_dim"], _config["goal_radius"])
        temp_buffer.init_buffer(data_path, cost_bound=_config["safety_threshold"])
        meta_buffer[task]=temp_buffer

    state_dim=_config["state_dim"]+_config["context_params"]["encode_dim"]
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

    # create policy
    # exp_policy = MetaCPQPolicy(
    #     actor,
    #     critic1,
    #     critic2,
    #     critic_c,
    #     vae_list,
    #     actor_optim,
    #     critic1_optim,
    #     critic2_optim,
    #     critic_c_optim,
    #     action_space=env_list[0].action_space,
    #     lagrange=None
    # )
    # exp_policy.load_state_dict(torch.load(_config["exp_policy_path"]))
    # exp_policy=copy.deepcopy(exp_policy)
    encoder=torch.load("run/"+_config['env_name']+"/context_encoder/"+_config["meta_params"][_config["meta_params"]["encoder_type"]+"_path"])
    policy = MetaCPQPolicy(
        actor,
        critic1,
        critic2,
        critic_c,
        vae_list,
        actor_optim,
        critic1_optim,
        critic2_optim,
        critic_c_optim,
        action_space=env_list[0].action_space,
        lagrange=None
    )
    policy.load_state_dict(torch.load(_config["policy_path"]))

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

    exp_context={0:[],1:[],2:[]}
    context=[[],[],[]]
    rw_ls={0:[],1:[],2:[]}
    cs_ls={0:[],1:[],2:[]}
    acu_cs_ls={0:[],1:[],2:[]}
    exp_trj_num=_config["exp_trj_num"]
    bootstrap_trj_num=_config["bootstrap_trj_num"]
    rnd_trj_num=_config["rnd_trj_num"]
    ep_l=_config["context_params"]["ep_length"]
    if _config["wo_exp"]:
        rnd_trj_num=_config["exp_trj_num"]+_config["bootstrap_trj_num"]+_config["rnd_trj_num"]
        exp_trj_num=0
        bootstrap_trj_num=0
    for goal in range(num_tasks-1,-1,-1):
        num_episodes=0
        obs = env_list[goal].reset()
        ep_reward=0
        ep_cost=0
        t_ep_l=0
        min_cost=300
        #z=np.random.randn(1,16)
        buffer=meta_buffer[num_episodes]
        whole_sequence=buffer.random_batch(300)
        cost_whole_context=torch.stack([torch.cat([whole_sequence["observations"], whole_sequence["actions"], 
                             whole_sequence["next_observations"], whole_sequence["costs"]], dim=-1)])
        z=encoder(cost_whole_context.squeeze().unsqueeze(0).to(device)).cpu().detach().numpy()
        temp_con=[]
        while num_episodes<exp_trj_num:
            action = policy.select_action(np.concatenate((obs.reshape(1,-1),z),axis=-1), deterministic=True)
            action = np.clip(action, env_list[goal].action_space.low, env_list[goal].action_space.high)
            next_obs, reward, terminal, info = env_list[goal].step(action.flatten())
            ep_reward+=reward
            t_ep_l+=1
            cost=info.get('cost', 0)
            con=torch.tensor(np.concatenate((obs.reshape(1,-1), action, next_obs.reshape(1,-1),np.array([cost]).reshape(1,-1)),axis=-1),dtype=torch.float32)
            exp_context[goal].append(con)
            temp_con.append(con)
            ep_cost+=cost
            if ep_cost>=30:
                terminal=True
            obs = next_obs
            if terminal:
                if _config["wo_iid"]:
                    context[goal].append(temp_con)
                else:
                    if context[goal]==[]:
                        context[goal].append(temp_con)
                        min_cost=ep_cost
                        min_reward=ep_reward
                        if ep_reward<=_config["iid_min_reward"]:
                            min_cost=300
                    elif ep_cost>min_cost and ep_cost<=_config["safety_threshold"]+5:
                        if ep_reward>min_reward:
                            context[goal].append(temp_con)
                            min_cost=ep_cost
                            min_reward=ep_reward
                        else:
                            context[goal].append(context[goal][-1])
                    elif ep_cost<min_cost and min_cost>_config["safety_threshold"]+5:
                        if ep_reward>_config["iid_min_reward"]:
                            #context[goal]+=temp_con
                            context[goal].append(temp_con)
                            min_cost=ep_cost
                            min_reward=ep_reward
                        else:
                            context[goal].append(context[goal][-1])
                    elif ep_cost <= min_cost:
                        if ep_reward>min_reward:
                            context[goal].append(temp_con)
                            min_cost=ep_cost
                            min_reward=ep_reward
                        else:
                            context[goal].append(context[goal][-1])
                    else:
                        context[goal].append(context[goal][-1])
                num_episodes +=1
                obs = env_list[goal].reset()
                rw_ls[goal].append(ep_reward)
                cs_ls[goal].append(ep_cost)
                if acu_cs_ls[goal]==[]:
                    acu_cs_ls[goal].append(ep_cost)
                else:
                    acu_cs_ls[goal].append(acu_cs_ls[goal][-1]+ep_cost)
                buffer=meta_buffer[num_episodes%num_tasks]
                whole_sequence=buffer.random_batch(300)
                cost_whole_context=torch.stack([torch.cat([whole_sequence["observations"], whole_sequence["actions"], 
                                    whole_sequence["next_observations"], whole_sequence["costs"]], dim=-1)])
                z=encoder(cost_whole_context.squeeze().unsqueeze(0).to(device)).cpu().detach().numpy()
                temp_con=[]
                ep_reward=0
                ep_cost=0
                t_ep_l=0
        num_episodes=0
        obs = env_list[goal].reset()
        ep_reward=0
        ep_cost=0
        t_ep_l=0
        z=np.zeros((1,16))
        #z=np.random.randn(1,16)
        temp_con=[]
        while num_episodes < rnd_trj_num:
            action = policy.select_action(np.concatenate((obs.reshape(1,-1),z),axis=-1), deterministic=True)
            action = np.clip(action, env_list[goal].action_space.low, env_list[goal].action_space.high)
            next_obs, reward, terminal, info = env_list[goal].step(action.flatten())
            ep_reward+=reward
            t_ep_l+=1
            cost=info.get('cost', 0)
            con=torch.tensor(np.concatenate((obs.reshape(1,-1), action, next_obs.reshape(1,-1),np.array([cost]).reshape(1,-1)),axis=-1),dtype=torch.float32)
            exp_context[goal].append(con)
            temp_con.append(con)
            ep_cost+=cost
            if ep_cost>=300:
                terminal=True
            obs = next_obs
            if terminal:
                if _config["wo_iid"]:
                    context[goal].append(temp_con)
                else:
                    if context[goal]==[]:
                        context[goal].append(temp_con)
                        min_cost=ep_cost
                        min_reward=ep_reward
                    elif ep_cost>min_cost and ep_cost<=_config["safety_threshold"]+25:
                        if ep_reward>min_reward:
                            context[goal].append(temp_con)
                            min_cost=ep_cost
                            min_reward=ep_reward
                        else:
                            context[goal].append(context[goal][-1])
                    elif ep_cost<min_cost and min_cost>_config["safety_threshold"]+25:
                        if ep_reward>_config["iid_min_reward"]:
                            #context[goal]+=temp_con
                            context[goal].append(temp_con)
                            min_cost=ep_cost
                            min_reward=ep_reward
                        else:
                            context[goal].append(context[goal][-1])
                    elif ep_cost <= min_cost:
                        if ep_reward>min_reward:
                            context[goal].append(temp_con)
                            min_cost=ep_cost
                            min_reward=ep_reward
                        else:
                            context[goal].append(context[goal][-1])
                    else:
                        context[goal].append(context[goal][-1])
                num_episodes +=1
                obs = env_list[goal].reset()
                rw_ls[goal].append(ep_reward)
                cs_ls[goal].append(ep_cost)
                if acu_cs_ls[goal]==[]:
                    acu_cs_ls[goal].append(ep_cost)
                else:
                    acu_cs_ls[goal].append(acu_cs_ls[goal][-1]+ep_cost)
                z=encoder(torch.stack(exp_context[goal][-t_ep_l:]).squeeze().unsqueeze(0).to(device)).cpu().detach().numpy()
                temp_con=[]
                ep_reward=0
                ep_cost=0
                t_ep_l=0
        
    print(rw_ls, cs_ls)
    total_rw_ls={0:[],1:[],2:[]}
    total_cs_ls={0:[],1:[],2:[]}
    for i in range(len(context[0])):
        rw_ls={0:[],1:[],2:[]}
        cs_ls={0:[],1:[],2:[]}
        for goal in range(num_tasks):
            num_episodes=0
            obs = env_list[goal].reset()
            ep_reward=0
            ep_cost=0
            if _config["wo_iid"] and not _config["wo_exp"]:
                z=encoder(torch.stack(context[goal][0]).squeeze().unsqueeze(0).to(device)).cpu().detach().numpy()
                for j in range(1,i+1):
                    z+=encoder(torch.stack(context[goal][j]).squeeze().unsqueeze(0).to(device)).cpu().detach().numpy()
                z=z/(i+1)
            else:
                z=encoder(torch.stack(context[goal][i]).squeeze().unsqueeze(0).to(device)).cpu().detach().numpy()
            while num_episodes < _config["eval_episodes"]:
                action = policy.select_action(np.concatenate((obs.reshape(1,-1),z),axis=-1), deterministic=True)
                action = np.clip(action, env_list[goal].action_space.low, env_list[goal].action_space.high)
                next_obs, reward, terminal, info = env_list[goal].step(action.flatten())
                ep_reward+=reward
                cost=info.get('cost', 0)
                ep_cost+=cost
                obs = next_obs
                if terminal:
                    num_episodes +=1
                    obs = env_list[goal].reset()
                    rw_ls[goal].append(ep_reward)
                    cs_ls[goal].append(ep_cost)
                    ep_reward=0
                    ep_cost=0
        for goal in range(num_tasks):
            logger.logkv("eval/episode_reward_mean"+str(goal), sum(rw_ls[goal])/len(rw_ls[goal]))
            logger.logkv("eval/episode_cost_mean"+str(goal), sum(cs_ls[goal])/len(cs_ls[goal]))
            logger.logkv("eval/accumulated_cost"+str(goal),acu_cs_ls[goal][i])
            total_rw_ls[goal].append(sum(rw_ls[goal])/len(rw_ls[goal]))
            total_cs_ls[goal].append(sum(cs_ls[goal])/len(cs_ls[goal]))
        logger.set_timestep(i)
        logger.dumpkvs()
    x=list(range(len(total_rw_ls[0])))
    x=[i+1 for i in x]
    for goal in range(num_tasks):
        plt.plot(x,total_rw_ls[goal],label="task_"+str(goal))
    plt.ylabel("reward")
    plt.xlabel("online trajectory num")
    plt.title("Online adaptation reward of our")
    plt.legend()
    plt.savefig(os.path.join(logger.result_dir, "adaptation reward"))
    plt.clf()
    for goal in range(num_tasks):
        plt.plot(x,total_cs_ls[goal],label="task_"+str(goal))
    plt.ylabel("cost")
    plt.xlabel("online trajectory num")
    plt.title("Online adaptation cost of our")
    plt.legend()
    plt.savefig(os.path.join(logger.result_dir, "adaptation cost"))
    np.save(os.path.join(logger.result_dir, "adaptation_context.npy"), np.array(context), allow_pickle=True)


@ex.main
def my_main():
    do_adaptation()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='ant-dir-safe')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--wo_iid', type=int, default=0)
    parser.add_argument('--wo_exp', type=int, default=0)
    parser.add_argument('--ood', type=int, default=0)
    parser.add_argument('--policy_path', type=str, default="run/cheetah-vel-safe/meta_cpq/mlp_attn/seed_1_timestamp_23-0729-204157/model/policy.pth")
    #parser.add_argument('--exp_policy_path', type=str, default="run/ant-walk-safe/meta_cpq_exp/mlp_attn/seed_0_timestamp_23-0829-184745/model/policy.pth")
    args = parser.parse_args()
    config_dict=read_json_dict("configs/"+args.task+".json")
    if args.seed>=0:
        config_dict["seed"]=args.seed
    config_dict["ood"]=args.ood
    if args.policy_path==None:
        raise ValueError("Policy path or exploration policy path should be given!")
    config_dict["policy_path"]=args.policy_path
    if args.ood and args.wo_iid and args.wo_exp:
        log_dirs = make_log_dirs(args.task, "task_generalization_wo_exp_iid", config_dict["seed"], config_dict)
    elif args.ood and args.wo_iid:
        log_dirs = make_log_dirs(args.task, "task_generalization_wo_iid", config_dict["seed"], config_dict)
    elif args.ood and args.wo_exp:
        log_dirs = make_log_dirs(args.task, "task_generalization_wo_exp", config_dict["seed"], config_dict)
    elif args.ood:
        log_dirs = make_log_dirs(args.task, "task_generalization", config_dict["seed"], config_dict)
    elif args.wo_iid and args.wo_exp:
        log_dirs = make_log_dirs(args.task, "online_adaptation_wo_exp_wo_iid", config_dict["seed"], config_dict)
    elif args.wo_iid:
        log_dirs = make_log_dirs(args.task, "online_adaptation_wo_iid", config_dict["seed"], config_dict)
    elif args.wo_exp:
        log_dirs = make_log_dirs(args.task, "online_adaptation_wo_exp", config_dict["seed"], config_dict)
    else:
        log_dirs = make_log_dirs(args.task, "online_adaptation", config_dict["seed"], config_dict)
    ex.add_config(config_dict)
    ex.add_config({"log_dirs":log_dirs})
    ex.add_config({"wo_exp":args.wo_exp})
    ex.add_config({"wo_iid":args.wo_iid})
    file_obs_path = os.path.join(log_dirs,"sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run()