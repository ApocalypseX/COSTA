import numpy as np

from rlkit.envs.mujoco.ant_multitask_base import MultitaskAntEnv
from rlkit.envs import register_env


@register_env('ant-goal-safe')
class AntGoalEnv(MultitaskAntEnv):

    def __init__(self, task={}, n_tasks=3, max_episode_steps=300, randomize_tasks=True, **kwargs):
        self.target = np.array([25.0, 25.0])
        super(AntGoalEnv, self).__init__(task, n_tasks, max_episode_steps, **kwargs)


    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))
        x_before=torso_xyz_before[0]
        y_before=torso_xyz_before[1]
        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        #wall = np.array([-5,5])
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        goal_dir=np.array([self.target[0]-x_before, self.target[1]-y_before])
        goal_dir=goal_dir/np.linalg.norm(goal_dir,2)
        dist_new = ((xposafter-self.target[0])**2+(yposafter-self.target[1])**2) ** 0.5
        forward_reward = np.dot((torso_velocity[:2]/self.dt), goal_dir)
        if dist_new<=1e-2:
            forward_reward = 5

        obj_cost=0.0
        if self._goal==0:
            if 10<=xposafter<=20 and 2<=yposafter<=20:
                obj_cost = 1.0
        elif self._goal==1:
            if 2<=xposafter<=20 and 10<=yposafter<=20:
                obj_cost = 1.0
        elif self._goal==2:
            obstacles=[np.array([5.0, 5.0]), np.array([5.0, 10.0]), np.array([10.0, 5.0]), 
                       np.array([12.5, 12.5]), np.array([10.0, 17.5]), np.array([17.5, 10.0]),
                       np.array([20.0, 20.0]), np.array([22.5, 25.0]), np.array([25.0, 22.5])]
            if xposafter<=0 or yposafter<=0 or xposafter>=27 or yposafter>=27:
                obj_cost=1.0
            else:
                for obs in obstacles:
                    if ((xposafter-obs[0])**2+(yposafter-obs[1])**2)**0.5<=0.75:
                        obj_cost=1.0
                        break
        else:
            assert False

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        done_cost = done*1.0
        cost = np.clip(obj_cost+done_cost, 0, 1)
        ob = self._get_obs()
        self._step += 1
        if self._step >= self._max_episode_steps:
            done = True
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            xposafter=xposafter,
            yposafter=yposafter,
            cost_obj = obj_cost,
            cost_done = done_cost,
            cost = cost,
        )
    
    def _get_obs(self):
        x = self.sim.data.qpos.flat[0]
        y = self.sim.data.qpos.flat[1]
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            [x/5],
            [y],
            #np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
    
    def reset(self):
        self._step = 0
        return super().reset()

    def sample_tasks(self, num_tasks):
        # if self.forward_backward:
        #     assert num_tasks == 2
        #     velocities = np.array([0., np.pi/2])
        # else:
        #     velocities = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
        velocities = np.arange(num_tasks)
        tasks = [{'goal': velocity} for velocity in velocities]
        return tasks
