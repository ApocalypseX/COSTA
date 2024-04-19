import numpy as np

from rlkit.envs.mujoco.ant_multitask_base import MultitaskAntEnv
from rlkit.envs import register_env


@register_env('ant-dir-safe')
class AntDirEnv(MultitaskAntEnv):

    def __init__(self, task={}, n_tasks=2, forward_backward=False, max_episode_steps=300, randomize_tasks=True, **kwargs):
        self.forward_backward = forward_backward
        super(AntDirEnv, self).__init__(task, n_tasks, max_episode_steps, **kwargs)


    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        if self._goal==0 or self._goal==1 or self._goal==2:
            direct = (np.cos(0.), np.sin(0.))
        elif self._goal==3 or self._goal==4:
            direct = (np.cos(-np.pi/6), np.sin(-np.pi/6))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)

        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]

        obj_cost=0.0
        if self._goal==0:
            if yposafter>3 or yposafter<-3:
                obj_cost = 1.0
        elif self._goal==1:
            if xposafter>=0 and xposafter<=10:
                if yposafter>4+xposafter or yposafter<-4+xposafter:
                    obj_cost = 1.0
            elif xposafter>10 and xposafter<=30:
                if yposafter>24-xposafter or yposafter<16-xposafter:
                    obj_cost = 1.0
            elif xposafter>30 and xposafter<=40:
                if yposafter>xposafter-36 or yposafter<xposafter-44:
                    obj_cost = 1.0
            else:
                if yposafter>3 or yposafter<-3:
                    obj_cost = 1.0
        elif self._goal==2:
            if xposafter>=0 and xposafter<=10:
                if yposafter>4-xposafter or yposafter<-4-xposafter:
                    obj_cost = 1.0
            elif xposafter>10 and xposafter<=30:
                if yposafter>xposafter-16 or yposafter<xposafter-24:
                    obj_cost = 1.0
            elif xposafter>30 and xposafter<=40:
                if yposafter>44-xposafter or yposafter<36-xposafter:
                    obj_cost = 1.0
            else:
                if yposafter>3 or yposafter<-3:
                    obj_cost = 1.0
        elif self._goal==3:
            if xposafter>=0 and xposafter<=20:
                if yposafter>4+0.5*xposafter or yposafter<-4+0.5*xposafter:
                    obj_cost = 1.0
            elif xposafter>20 and xposafter<=40:
                if yposafter>-0.5*xposafter+24 or yposafter<-0.5*xposafter+16:
                    obj_cost = 1.0
            else:
                if yposafter>3 or yposafter<-3:
                    obj_cost = 1.0
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
            torso_velocity=torso_velocity,
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
