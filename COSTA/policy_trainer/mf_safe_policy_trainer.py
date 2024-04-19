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
class MFSafePolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        buffer: SimpleSafeReplayBuffer,
        logger: Logger,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        sequence_batch_size: int = 3,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        safety_bound: float = 25.0,
        use_state_augmentation: bool = True,
        use_sequence_batch: bool = True
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
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
                if self.use_sequence_batch:
                    batch = self.buffer.random_sequence(self._sequence_batch_size)
                else:
                    batch = self.buffer.random_batch(self._batch_size)
                loss = self.policy.learn(batch, e)
                pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # evaluate current policy
            eval_info = self._evaluate()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            ep_cost_mean, ep_cost_std = np.mean(eval_info["eval/episode_cost"]), np.std(eval_info["eval/episode_cost"])
            last_10_performance.append(ep_reward_mean)
            self.logger.logkv("eval/episode_reward", ep_reward_mean)
            self.logger.logkv("eval/episode_reward_std", ep_reward_std)
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.logkv("eval/episode_cost", ep_cost_mean)
            self.logger.logkv("eval/episode_cost_std", ep_cost_std)
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs()
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        #assert self.eval_env._goal==1
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        episode_cost = 0
        cost_state = self.safety_bound

        while num_episodes < self._eval_episodes:
            if self.use_state_augmentation:
                obs=np.append(obs, np.array([cost_state/10]), axis=-1)
            action = self.policy.select_action(obs.reshape(1,-1), deterministic=True)
            action = np.clip(action, self.eval_env.action_space.low, self.eval_env.action_space.high)
            next_obs, reward, terminal, info = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_cost += info.get('cost', 0)
            cost_state -= info.get('cost', 0)
            cost_state = max(cost_state, 0.0)
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length, "episode_cost": episode_cost}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                episode_cost = 0
                cost_state = self.safety_bound
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "eval/episode_cost": [ep_info["episode_cost"] for ep_info in eval_ep_info_buffer],
        }
