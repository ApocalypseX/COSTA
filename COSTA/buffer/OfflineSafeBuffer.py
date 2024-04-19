import numpy as np
import glob
import os
import abc
import torch

#preprocess for quicker loader
def init_buffer(n_trj, data_dir, task_name):
    train_trj_paths = []
    # trj entry format: [obs, action, reward, cost, new_obs]
    for n in range(n_trj):
        train_trj_paths += glob.glob(os.path.join(data_dir, task_name, "trj_eval%d_epoch*.npy" %(n)))
    
    obs_train_lst = []
    action_train_lst = []
    reward_train_lst = []
    cost_train_lst = []
    next_obs_train_lst = []
    terminal_train_lst = []
    
    for train_path in train_trj_paths:
        trj_npy = np.load(train_path, allow_pickle=True)
        obs_train_lst += list(trj_npy[:, 0])
        action_train_lst += list(trj_npy[:, 1])
        reward_train_lst += list(trj_npy[:, 2])
        cost_train_lst += list(trj_npy[:, 3])
        next_obs_train_lst += list(trj_npy[:, 4])
        terminal = [0 for _ in range(trj_npy.shape[0])]
        terminal[-1] = 1
        terminal_train_lst += terminal
    np.savez(os.path.join(data_dir, task_name, "offline_buffer_with_{}_trj_per_epoch".format(n_trj)), obs=np.array(obs_train_lst),
            action=np.array(action_train_lst), reward=np.array(reward_train_lst), cost=np.array(cost_train_lst), 
             next_obs=np.array(next_obs_train_lst), terminal=np.array(terminal_train_lst))

class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """
    A class used to save and replay data.
    """

    @abc.abstractmethod
    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        """
        Add a transition tuple.
        """
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                # agent_info,
                # env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            # path["agent_infos"],
            # path["env_infos"],
        )):
            self.add_sample(
                obs,
                action,
                reward,
                terminal,
                next_obs,
                # agent_info=agent_info,
                # env_info=env_info,
                **{'env_info': {}}
            )
        self.terminate_episode()

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass


#ReplayBuffer for safe data
class SimpleSafeReplayBuffer(ReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim, goal_radius=0.2, require_false_buffer=False
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._costs = np.zeros((max_replay_buffer_size, 1))
        self._costs_state = np.zeros((max_replay_buffer_size, 1))
        self._sparse_rewards = np.zeros((max_replay_buffer_size, 1))
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self.goal_radius = goal_radius
        self.clear()
        self.require_false_buffer = require_false_buffer
    
    def init_buffer(self, path=None, n_trj=10, data_dir=None, task_name=None, cost_bound=25.0):
        if path is not None:
            whole_data = np.load(path)
            obs_train_lst = list(whole_data["obs"])
            action_train_lst = list(whole_data["action"])
            reward_train_lst = list(whole_data["reward"])
            cost_train_lst = list(whole_data["cost"])
            next_obs_train_lst = list(whole_data["next_obs"])
            terminal_train_lst = list(whole_data["terminal"])
            cost_state_lst = []
            cost_left = cost_bound
            for i in range(len(cost_train_lst)):
                cost_state_lst.append(np.maximum(np.array(cost_left), np.array(0.0)))
                cost_left-=cost_train_lst[i]
                if terminal_train_lst[i] == 1:
                    cost_left = cost_bound
        else:
            train_trj_paths = []
            # trj entry format: [obs, action, reward, cost, new_obs]
            for n in range(n_trj):
                train_trj_paths += glob.glob(os.path.join(data_dir, task_name, "trj_eval%d_epoch*.npy" %(n)))
            obs_train_lst = []
            action_train_lst = []
            reward_train_lst = []
            cost_train_lst = []
            next_obs_train_lst = []
            terminal_train_lst = []

            for train_path in train_trj_paths:
                #print(train_path)
                trj_npy = np.load(train_path, allow_pickle=True)
                #print(trj_npy[0])
                obs_train_lst += list(trj_npy[:, 0])
                action_train_lst += list(trj_npy[:, 1])
                reward_train_lst += list(trj_npy[:, 2])
                cost_train_lst += list(trj_npy[:, 3])
                next_obs_train_lst += list(trj_npy[:, 4])
                terminal = [0 for _ in range(trj_npy.shape[0])]
                terminal[-1] = 1
                terminal_train_lst += terminal
        for i, (
                obs,
                action,
                reward,
                cost,
                next_obs,
                terminal,
                cost_state
        ) in enumerate(zip(
            obs_train_lst,
            action_train_lst,
            reward_train_lst,
            cost_train_lst,
            next_obs_train_lst,
            terminal_train_lst,
            cost_state_lst
        )):
            self.add_sample(
                obs,
                np.squeeze(action),
                reward,
                cost,
                terminal,
                next_obs,
                cost_state,
                **{'env_info': {}},
            )
            
    def add_sample(self, observation, action, reward, cost, terminal,
                   next_observation, cost_state=0.0, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._costs[self._top] = cost
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        self._costs_state[self._top] = cost_state/10

        if reward >= self.goal_radius:
            sparse_reward = (reward - self.goal_radius) * (1/abs(self.goal_radius))
            self._sparse_rewards[self._top] = kwargs['env_info'].get('sparse_reward', sparse_reward)
        else:
            self._sparse_rewards[self._top] = kwargs['env_info'].get('sparse_reward', 0)
        self._advance()
        if terminal:
            self.terminate_episode()

    def terminate_episode(self):
        # store the episode beginning once the episode is over
        # n.b. allows last episode to loop but whatever
        self._episode_starts.append(self._cur_episode_start)
        self._cur_episode_start = self._top

    def size(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._cur_episode_start = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def sample_data(self, indices):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not self.require_false_buffer:
            return dict(
                observations=torch.tensor(self._observations[indices], dtype=torch.float32).to(device),
                actions=torch.tensor(self._actions[indices], dtype=torch.float32).to(device),
                rewards=torch.tensor(self._rewards[indices], dtype=torch.float32).to(device),
                costs=torch.tensor(self._costs[indices], dtype=torch.float32).to(device),
                terminals=torch.tensor(self._terminals[indices], dtype=torch.float32).to(device),
                next_observations=torch.tensor(self._next_obs[indices],dtype=torch.float32).to(device),
                sparse_rewards=torch.tensor(self._sparse_rewards[indices], dtype=torch.float32).to(device),
                costs_state=torch.tensor(self._costs_state[indices], dtype=torch.float32).to(device)
            )
        else:
            return dict(
                observations=torch.tensor(self._observations[indices], dtype=torch.float32).to(device),
                actions=torch.tensor(self._actions[indices], dtype=torch.float32).to(device),
                rewards=torch.tensor(self._rewards[indices], dtype=torch.float32).to(device),
                costs=torch.tensor(self._costs[indices], dtype=torch.float32).to(device),
                terminals=torch.tensor(self._terminals[indices], dtype=torch.float32).to(device),
                next_observations=torch.tensor(self._next_obs[indices],dtype=torch.float32).to(device)
            )
    
    def sample_all(self):
        #for supervised learning like learning VAE or dynamic model
        if not self.require_false_buffer:
            return dict(
                observations=self._observations[:self._size].copy(),
                actions=self._actions[:self._size].copy(),
                rewards=self._rewards[:self._size].copy(),
                costs=self._costs[:self._size].copy(),
                terminals=self._terminals[:self._size].copy(),
                next_observations=self._next_obs[:self._size].copy(),
                sparse_rewards=self._sparse_rewards[:self._size].copy(),
                costs_state=self._costs_state[:self._size].copy()
            )
        else:
            return dict(
                observations=self._observations[:self._size].copy(),
                actions=self._actions[:self._size].copy(),
                rewards=self._rewards[:self._size].copy(),
                costs=self._costs[:self._size].copy(),
                terminals=self._terminals[:self._size].copy(),
                next_observations=self._next_obs[:self._size].copy()
            )

    def random_batch(self, batch_size):
        ''' batch of unordered transitions '''
        assert self._size > 0
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices)

    def random_sequence(self, batch_size, ep_length=300, require_first_sequence=False):
        ''' batch of trajectories '''
        # take random batch of trajectories 
        # batch size 2 means we sample 2 whole trajectories and return with batch shape
        # for example if we set batch size as 2, and episode length is 300, then we get a batch with 600 samples
        # the reason we need this function is that when we do meta learning for safe, there may be only some of transitions in
        # an episode is useful, so we need some whole episodes as a batch in order not to miss these useful transitions
        i = 0
        indices = []
        first_indices =[]
        while i < batch_size:
            start = np.random.choice(self._episode_starts[:-1])
            pos_idx = self._episode_starts.index(start)
            if require_first_sequence and len(first_indices)==0:
                if len(list(range(start, self._episode_starts[pos_idx + 1])))==ep_length:
                    first_indices += list(range(start, self._episode_starts[pos_idx + 1]))
                else:
                    continue
            indices += list(range(start, self._episode_starts[pos_idx + 1]))
            i += 1
        if len(indices)<batch_size*ep_length:
            indices+=list(np.random.randint(0, self._size, batch_size*ep_length-len(indices)))
        if require_first_sequence:
            return self.sample_data(indices), self.sample_data(first_indices)
        return self.sample_data(indices)

    def num_steps_can_sample(self):
        return self._size