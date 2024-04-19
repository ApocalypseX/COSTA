#!/usr/bin/env python
import gym 
import os
import json
import safety_gym
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs import ENVS
from configs.default import default_config

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

def main(task, algo, seed, cpu):

    # Verify experiment
    #robot_list = ['point', 'car', 'doggo']
    task_list = ['ant-dir-safe']
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']
    goal_list = [0,1,2]

    algo = algo.lower()
    #task = task.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"


    # Hyperparameters
    exp_name = algo + '_' + task + '_meta_'+str(goal_list)
    num_steps = 3e7
    steps_per_epoch = 6000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = 25

    # Fork for parallelizing
    mpi_fork(cpu)

    # Prepare Logger
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    algo = eval('safe_rl.'+algo)
    variant = default_config
    config="configs/"+task+".json"
    #cwd = os.getcwd()
    #files = os.listdir(cwd)
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = 0
    envs=[]
    for goal in goal_list:
        env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
        env.seed(seed)
        env.reset_task(goal)
        envs.append(env)
    
    for env in envs:
        print(env._goal)

    algo(env_fn=None,
         ac_kwargs=dict(
             hidden_sizes=(256, 256),
            ),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         target_kl=target_kl,
         cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs,
         env=None,
         max_ep_len=300,
         env_ls=envs,
         env_name=task,
         )



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='ant-dir-safe')
    parser.add_argument('--algo', type=str, default='cpo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--goal', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()
    #exp_name = args.exp_name if not(args.exp_name=='') else None
    main(args.task, args.algo, args.seed, args.cpu)