import numpy as np

from rlkit.envs.mujoco.ant_multitask_base import MultitaskAntEnv
from rlkit.envs import register_env


@register_env('ant-circle-safe')
class AntCircleEnv(MultitaskAntEnv):

    def __init__(self, task={}, n_tasks=3, max_episode_steps=300, randomize_tasks=True, **kwargs):
        super(AntCircleEnv, self).__init__(task, n_tasks, max_episode_steps, **kwargs)


    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_v, y_v = xy_velocity

        x_pos, y_pos = xy_position_after[0], xy_position_after[1]
        d_xy = np.linalg.norm(xy_position_after, ord=2)
        d_o = 10
        lim = 6

        # Get reward
        circle_reward = (- x_v * y_pos + y_v * x_pos) / (1 + np.abs(d_xy - d_o))

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = circle_reward - ctrl_cost - contact_cost + survive_reward

        if self._goal==0:
            obj_cost = np.float(np.abs(x_pos) > lim)
        elif self._goal==1:
            obj_cost = np.float(np.abs(y_pos) > lim)
        elif self._goal==2:
            obj_cost = np.float(np.abs(y_pos-x_pos) > lim*np.sqrt(2))
        else:
            assert False

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
            reward_circle=circle_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            xposafter=x_pos,
            yposafter=y_pos,
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
