import argparse
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from nets import MLP
from modules import ActorProb, Critic, TanhDiagGaussian
from utils.logger import Logger, make_log_dirs
from policy_trainer import MFMetaSafePolicyTrainer
from policy import PEARLCPQPolicy
from sacred import Experiment
from sacred.observers import FileStorageObserver
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs import ENVS
from model.vae_model import *
from utils.common import seed_torch, read_json_dict
from nets.encoder import *
import matplotlib.pyplot as plt

ex = Experiment()

def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared

@ex.capture
def do_adaptation(_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed=_config["seed"]
    # create env and dataset
    num_tasks=_config["num_tasks"]
    env_list=[]
    vae_list = []
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
    encoder = MLPUDEncoder(input_dim=2*config_dict["state_dim"]+config_dict["action_dim"]+1, 
                       hidden_dims=config_dict["context_params"]["hidden_dims"], output_dim=config_dict["context_params"]["encode_dim"])
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=1e-3)

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
    ep_l=_config["context_params"]["ep_length"]
    for goal in range(num_tasks):
        num_episodes=0
        obs = env_list[goal].reset()
        ep_reward=0
        ep_cost=0
        min_cost=300
        z=np.zeros((1,16))
        temp_con=[]
        while num_episodes < _config["exp_trj_num"]+_config["rnd_trj_num"]+_config["bootstrap_trj_num"]:
            action = policy.select_action(np.concatenate((obs.reshape(1,-1),z),axis=-1), deterministic=False)
            action = np.clip(action, env_list[goal].action_space.low, env_list[goal].action_space.high)
            next_obs, reward, terminal, info = env_list[goal].step(action.flatten())
            ep_reward+=reward
            cost=info.get('cost', 0)
            con=torch.tensor(np.concatenate((obs.reshape(1,-1), action, next_obs.reshape(1,-1),np.array([cost]).reshape(1,-1)),axis=-1),dtype=torch.float32)
            exp_context[goal].append(con)
            temp_con.append(con)
            ep_cost+=cost
            obs = next_obs
            if terminal:
                context[goal].append(temp_con)
                num_episodes +=1
                obs = env_list[goal].reset()
                rw_ls[goal].append(ep_reward)
                cs_ls[goal].append(ep_cost)
                mean, sigma_squared=policy.encoder(torch.stack(exp_context[goal][-ep_l:]).squeeze().unsqueeze(0).to(device))
                z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mean), torch.unbind(sigma_squared))]
                z_means = torch.stack([p[0] for p in z_params])
                z_vars = torch.stack([p[1] for p in z_params])
                posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(z_means), torch.unbind(z_vars))]
                z = [d.rsample() for d in posteriors]
                z = torch.stack(z).cpu().detach().numpy()
                temp_con=[]
                ep_reward=0
                ep_cost=0
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
            mean, sigma_squared=policy.encoder(torch.stack(context[goal][i]).squeeze().unsqueeze(0).to(device))
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mean), torch.unbind(sigma_squared))]
            z_means = torch.stack([p[0] for p in z_params])
            z_vars = torch.stack([p[1] for p in z_params])
            posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(z_means), torch.unbind(z_vars))]
            z = [d.rsample() for d in posteriors]
            z = torch.stack(z).cpu().detach().numpy()
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
    plt.title("Online adaptation reward of pearl")
    plt.legend()
    plt.savefig(os.path.join(logger.result_dir, "adaptation reward"))
    plt.clf()
    for goal in range(num_tasks):
        plt.plot(x,total_cs_ls[goal],label="task_"+str(goal))
    plt.ylabel("cost")
    plt.xlabel("online trajectory num")
    plt.title("Online adaptation cost of pearl")
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
    parser.add_argument('--ood', type=int, default=0)
    parser.add_argument('--policy_path', type=str, default="run/cheetah-vel-safe/pearl_cpq/seed_0_timestamp_23-0730-101355/model/policy.pth")
    args = parser.parse_args()
    config_dict=read_json_dict("configs/"+args.task+".json")
    config_dict["ood"]=args.ood
    if args.seed>=0:
        config_dict["seed"]=args.seed
    seed_torch(config_dict["seed"])
    if args.policy_path==None :
        raise ValueError("Policy path or exploration policy path should be given!")
    config_dict["policy_path"]=args.policy_path
    if args.ood:
        log_dirs = make_log_dirs(args.task, "task_generalization_pearl", config_dict["seed"], config_dict)
    else:
        log_dirs = make_log_dirs(args.task, "online_adaptation_pearl", config_dict["seed"], config_dict)
    ex.add_config(config_dict)
    ex.add_config({"log_dirs":log_dirs})
    file_obs_path = os.path.join(log_dirs,"sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run()