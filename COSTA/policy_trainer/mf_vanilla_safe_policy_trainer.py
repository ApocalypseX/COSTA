import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List
from tqdm import tqdm
from collections import deque
from buffer import SimpleSafeReplayBuffer
from utils.logger import Logger
from policy import BasePolicy


# model-free policy trainer
class MFVanillaSafePolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        encoder,
        eval_env_list: list,
        num_tasks,
        buffer: dict,
        logger: Logger,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        sequence_batch_size: int = 3,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        safety_bound: float = 25.0,
        use_state_augmentation: bool = False,
        use_sequence_batch: bool = False,
        context_sequence_num: int = 1,
        is_focal: bool = False
    ) -> None:
        self.policy = policy
        self.eval_env_list = eval_env_list
        self.num_tasks = num_tasks
        self.buffer = buffer
        self.logger = logger

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._sequence_batch_size = sequence_batch_size
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

        self.use_state_augmentation = use_state_augmentation
        self.safety_bound = safety_bound

        self.use_sequence_batch = use_sequence_batch
        self.encoder=encoder
        self.context_sequence_num=context_sequence_num

        self.is_focal=is_focal
    
    def sample_context(self, task, train=False):
        cost_context=[]
        cost_context_first_sequence=[]
        buffer=self.buffer[task]
        if train==True:
            s_num=1
        else:
            s_num=self.context_sequence_num
        for j in range(s_num):
            if self.is_focal:
                whole_sequence=buffer.random_batch(self.eval_env_list[0]._max_episode_steps)
                cost_whole_context=torch.cat([whole_sequence["observations"], whole_sequence["actions"], 
                                            whole_sequence["next_observations"], whole_sequence["costs"]], dim=-1)
                cost_context.append(cost_whole_context)
            else:
                whole_sequence, first_sequence=buffer.random_sequence(1, require_first_sequence=True)
                cost_whole_context=torch.cat([whole_sequence["observations"], whole_sequence["actions"], 
                                            whole_sequence["next_observations"], whole_sequence["costs"]], dim=-1)
                cost_first_context=torch.cat([first_sequence["observations"], first_sequence["actions"], 
                                            first_sequence["next_observations"], first_sequence["costs"]], dim=-1)
                cost_context.append(cost_whole_context)
                cost_context_first_sequence.append(cost_first_context)
        cost_context=torch.stack(cost_context, dim=0)
        if self.is_focal:
            return cost_context, None
        cost_context_first_sequence=torch.stack(cost_context_first_sequence, dim=0)
        return cost_context, cost_context_first_sequence

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # train loop
        for e in range(1, self._epoch + 1):

            self.policy.train()

            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
            for it in pbar:
                #batch = self.buffer.random_batch(self._batch_size)
                res={}
                for task in range(self.num_tasks):
                    if self.use_sequence_batch:
                        batch = self.buffer[task].random_sequence(self._sequence_batch_size)
                    else:
                        batch = self.buffer[task].random_batch(self._batch_size)
                    loss = self.policy.learn(batch, None, task, e)
                    res = dict(list(res.items())+list(loss.items()))
                pbar.set_postfix(**res)

                for k, v in res.items():
                    self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # evaluate current policy
            eval_info = self._evaluate()
            total_ep_reward_mean=[]
            total_ep_cost_mean=[]
            for task in range(self.num_tasks):
                ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"+"_"+str(task)]), np.std(eval_info["eval/episode_reward"+"_"+str(task)])
                ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"+"_"+str(task)]), np.std(eval_info["eval/episode_length"+"_"+str(task)])
                ep_cost_mean, ep_cost_std = np.mean(eval_info["eval/episode_cost"+"_"+str(task)]), np.std(eval_info["eval/episode_cost"+"_"+str(task)])
                total_ep_reward_mean.append(ep_reward_mean)
                total_ep_cost_mean.append(ep_cost_mean)
                self.logger.logkv("eval/episode_reward"+"_"+str(task), ep_reward_mean)
                #self.logger.logkv("eval/episode_reward_std"+"_"+str(task), ep_reward_std)
                self.logger.logkv("eval/episode_length"+"_"+str(task), ep_length_mean)
                #self.logger.logkv("eval/episode_length_std"+"_"+str(task), ep_length_std)
                self.logger.logkv("eval/episode_cost"+"_"+str(task), ep_cost_mean)
                #self.logger.logkv("eval/episode_cost_std"+"_"+str(task), ep_cost_std)
            ep_reward_mean=sum(total_ep_reward_mean)/len(total_ep_reward_mean)
            ep_cost_mean=sum(total_ep_cost_mean)/len(total_ep_cost_mean)
            self.logger.logkv("eval/episode_reward_mean", ep_reward_mean)
            self.logger.logkv("eval/episode_cost_mean", ep_cost_mean)
            last_10_performance.append(ep_reward_mean)
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs()
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self) -> Dict[str, List[float]]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.eval()
        eval_ep_info_buffer = []
        for task in range(self.num_tasks):
            env=self.eval_env_list[task]
            obs = env.reset()
            num_episodes = 0
            episode_reward, episode_length = 0, 0
            episode_cost = 0
            cost_state = self.safety_bound

            while num_episodes < self._eval_episodes:
                if self.use_state_augmentation:
                    obs=np.append(obs, np.array([cost_state/10]), axis=-1)
                #obs_input=torch.cat([torch.tensor(obs.reshape(1,-1), dtype=torch.float32).to(device), context_encoding], dim=-1)
                obs_input=obs
                action = self.policy.select_action(obs_input, deterministic=True)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                next_obs, reward, terminal, info = env.step(action.flatten())
                episode_reward += reward
                episode_cost += info.get('cost', 0)
                cost_state -= info.get('cost', 0)
                cost_state = max(cost_state, 0.0)
                episode_length += 1

                obs = next_obs

                if terminal:
                    eval_ep_info_buffer.append(
                        {"episode_reward"+"_"+str(task): episode_reward, "episode_length"+"_"+str(task): episode_length, "episode_cost"+"_"+str(task): episode_cost}
                    )
                    num_episodes +=1
                    episode_reward, episode_length = 0, 0
                    episode_cost = 0
                    cost_state = self.safety_bound
                    obs = env.reset()
        res={}
        for task in range(self.num_tasks):
            res["eval/episode_reward"+"_"+str(task)]=[]
            res["eval/episode_length"+"_"+str(task)]=[]
            res["eval/episode_cost"+"_"+str(task)]=[]
        for ep_info in eval_ep_info_buffer:
            for task in range(self.num_tasks):
                if str(task) in list(ep_info.keys())[0]:
                    res["eval/episode_reward"+"_"+str(task)].append(ep_info["episode_reward"+"_"+str(task)])
                    res["eval/episode_length"+"_"+str(task)].append(ep_info["episode_length"+"_"+str(task)])
                    res["eval/episode_cost"+"_"+str(task)].append(ep_info["episode_cost"+"_"+str(task)])
        return res