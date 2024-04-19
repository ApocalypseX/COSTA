import numpy as np
import torch
import torch.nn as nn
import gym
from copy import deepcopy

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from policy import SACPolicy
from lagrange.lagrange import Lagrange


class MetaCPQPolicy(SACPolicy):
    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        critic_c: nn.Module, #the cost critic
        vae_list, #vae should be pre-trained
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        critic_c_optim: torch.optim.Optimizer, #the cost critic optimizer
        action_space: gym.spaces.Space,
        lagrange,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        cql_weight: float = 1.0,
        cpq_weight: float = 1.0,
        temperature: float = 1.0,
        max_q_backup: bool = False,
        deterministic_backup: bool = True,
        with_lagrange: bool = True,
        use_vae: bool = True,
        train_cpq_alpha = True,  #with_lagrange_cpq
        lagrange_threshold: float = 10.0,
        cql_alpha_lr: float = 1e-4,
        safety_threshold: float = 25.0,
        kl_threshold: float = 16.0,
        cpq_alpha_lr: float = 1e-4,
        num_repeart_actions: int = 10,
        policy_train: str = "sac",
        use_state_augmentation: bool = True,
        use_safety_lagrange: bool = False,
        safety_lagrange_pid: bool = False,
        warm_up_epoch: int = 50,
        use_conservative_reward_loss: bool = True,
        use_conservative_cost_loss: bool = False,
        lgr_lower_bound: float = 1.0,
        lgr_upper_bound: float = 10.0
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self.critic_c, self.critic_c_old = critic_c, deepcopy(critic_c)
        self.critic_c_old.eval()
        self.critic_c_optim = critic_c_optim

        self.vae_list = vae_list
        if self.vae_list is not None:
            for vae in self.vae_list:
                vae.eval()
        self.use_vae = use_vae

        self.action_space = action_space
        self._cql_weight = cql_weight
        self._temperature = temperature
        self._max_q_backup = max_q_backup
        self._deterministic_backup = deterministic_backup
        self._with_lagrange = with_lagrange
        self._lagrange_threshold = lagrange_threshold
        self.safety_threshold = safety_threshold
        self._lagrange_safety_threshold = safety_threshold * 1.5
  
        self.cql_log_alpha = torch.zeros(1, requires_grad=True, device=self.actor.device)
        self.cql_alpha_optim = torch.optim.Adam([self.cql_log_alpha], lr=cql_alpha_lr)

        self.train_cpq_alpha = train_cpq_alpha
        self.cpq_log_alpha = torch.zeros(1, requires_grad=True, device=self.actor.device)
        self.cpq_alpha_optim = torch.optim.Adam([self.cpq_log_alpha], lr=cpq_alpha_lr)
        self.cpq_weight = cpq_weight 

        self._num_repeat_actions = num_repeart_actions

        self.policy_train = policy_train
        self.kl_threshold = kl_threshold

        self.use_state_augmentation = use_state_augmentation

        self.use_safety_lagrange = use_safety_lagrange
        self.lagrange = lagrange
        if safety_lagrange_pid:
            self.lagrange_type = "pid"
        else:
            self.lagrange_type = "common"
        
        self.warm_up_epoch = warm_up_epoch

        self.use_conservative_reward_loss = use_conservative_reward_loss
        self.use_conservative_cost_loss = use_conservative_cost_loss

        self.lgr_lower_bound = lgr_lower_bound
        self.lgr_upper_bound = lgr_upper_bound

    def calc_pi_values(
        self,
        obs_pi: torch.Tensor,
        obs_to_pred: torch.Tensor,
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        act, log_prob = self.actforward(obs_pi)

        q1 = self.critic1(obs_to_pred, act)
        q2 = self.critic2(obs_to_pred, act)

        if not deterministic:
            return q1 - log_prob.detach(), q2 - log_prob.detach()
        else:
            return q1, q2 


    def calc_random_values(
        self,
        obs: torch.Tensor,
        random_act: torch.Tensor,
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self.critic1(obs, random_act)
        q2 = self.critic2(obs, random_act)

        log_prob1 = np.log(0.5**random_act.shape[-1])
        log_prob2 = np.log(0.5**random_act.shape[-1])

        if not deterministic:
            return q1 - log_prob1.detach(), q2 - log_prob2.detach()
        else:
            return q1, q2 
    
    def calc_pi_values_c(
        self,
        obs_pi: torch.Tensor,
        obs_to_pred: torch.Tensor,
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        act, log_prob = self.actforward(obs_pi)

        q_c = self.critic_c(obs_to_pred, act)

        if not deterministic:
            return q_c - log_prob.detach()
        else:
            return q_c 


    def calc_random_values_c(
        self,
        obs: torch.Tensor,
        random_act: torch.Tensor,
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_c = self.critic_c(obs, random_act)

        log_prob = np.log(0.5**random_act.shape[-1])

        if not deterministic:
            return q_c - log_prob.detach()
        else:
            return q_c
    
    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.critic_c.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.critic_c.eval()

    def learn(self, batch: Dict, context, task, e=0) -> Dict[str, float]:
        org_task=task
        if task==-1:
            org_task=0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        obss, actions, next_obss, rewards, costs, terminals  = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["costs"], batch["terminals"]
        if self.use_state_augmentation:
            costs_state = batch["costs_state"]
        batch_size = obss.shape[0]
        org_context=context
        context=context.repeat(batch_size,1).to(device)
        if self.use_state_augmentation:
            old_obss = deepcopy(obss)
            old_next_obss = deepcopy(next_obss)
            obss = torch.cat([obss, costs_state], dim=-1)
            next_costs_state = torch.max(costs_state-costs/10, torch.tensor(0.0).to(device))
            next_obss = torch.cat([next_obss, next_costs_state], dim=-1)
        
        # update actor
        obss_with_context=torch.cat([obss, context], dim=-1)
        next_obss_with_context=torch.cat([next_obss, context], dim=-1)
        # obss_with_context=obss
        # next_obss_with_context=next_obss
        if self.policy_train == "sac":
            a, log_probs = self.actforward(obss_with_context)
            #这里之所以要将context一同输入给reward critic是因为cpq中的reward critic也是与cost有关的
            q1a, q2a = self.critic1(obss_with_context, a), self.critic2(obss_with_context, a)
            if self.use_safety_lagrange:
                qca = self.critic_c(obss_with_context, a)
                if self.lagrange_type == "common":
                    lgr=self.lagrange[task].lagrangian_multiplier.item()
                    if lgr<self.lgr_lower_bound:
                        lgr=self.lgr_lower_bound
                    elif lgr>self.lgr_upper_bound:
                        lgr=self.lgr_upper_bound
                    actor_loss = (self._alpha * log_probs - (torch.min(q1a, q2a) - lgr * qca)).mean()
                    if e > self.warm_up_epoch:
                        self.lagrange[task].update_lagrange_multiplier(qca.mean().item())
                elif self.lagrange_type == "pid":
                    actor_loss = (self._alpha * log_probs - (torch.min(q1a, q2a) - lgr * qca)).mean()
                    if e > self.warm_up_epoch:
                        self.lagrange[task].pid_update(qca.mean().item())
                else:
                    raise ValueError("Not implemented yet.")
            else:
                actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            if self._is_auto_alpha:
                log_probs = log_probs.detach() + self._target_entropy
                alpha_loss = -(self._log_alpha * log_probs).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self._alpha = self._log_alpha.detach().exp()
        elif self.policy_train == "dpg":
            a, log_probs = self.actforward(obss_with_context)
            q1a, q2a = self.critic1(obss_with_context, a), self.critic2(obss_with_context, a)
            if self.use_safety_lagrange:
                qca = self.critic_c(obss_with_context, a)
                if self.lagrange_type == "common":
                    lgr=self.lagrange[task].lagrangian_multiplier.item()
                    if lgr<self.lgr_lower_bound:
                        lgr=self.lgr_lower_bound
                    elif lgr>self.lgr_upper_bound:
                        lgr=self.lgr_upper_bound
                    actor_loss = ( - (torch.min(q1a, q2a) - lgr * qca)).mean()
                    if e > self.warm_up_epoch:
                        self.lagrange[task].update_lagrange_multiplier(qca.mean().item())
                elif self.lagrange_type == "pid":
                    actor_loss = ( - (torch.min(q1a, q2a) - lgr * qca)).mean()
                    if e > self.warm_up_epoch:
                        self.lagrange[task].update_lagrange_multiplier(qca.mean().item())
                else:
                    raise ValueError("Not implemented yet.")
            else:
                actor_loss = ( - torch.min(q1a, q2a)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
        
        # compute td error
        if self._max_q_backup:
            with torch.no_grad():
                tmp_next_obss = next_obss_with_context.unsqueeze(1) \
                    .repeat(1, self._num_repeat_actions, 1) \
                    .view(batch_size * self._num_repeat_actions, next_obss_with_context.shape[-1])
                tmp_next_actions, _ = self.actforward(tmp_next_obss)
                tmp_next_q1 = self.critic1_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                tmp_next_q2 = self.critic2_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                next_q = torch.min(tmp_next_q1, tmp_next_q2)
                #cpq indicator
                target_Qc = self.critic_c_old(next_obss_with_context, next_actions)
                if self.use_state_augmentation:
                    weight = torch.where(target_Qc > next_costs_state * 10, torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
                else:
                    weight = torch.where(target_Qc > self.safety_threshold, torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
        else:
            with torch.no_grad():
                next_actions, next_log_probs = self.actforward(next_obss_with_context)
                next_q = torch.min(
                    self.critic1_old(next_obss_with_context, next_actions),
                    self.critic2_old(next_obss_with_context, next_actions)
                )
                if not self._deterministic_backup:
                    next_q -= self._alpha * next_log_probs
                #cpq indicator
                target_Qc = self.critic_c_old(next_obss_with_context, next_actions)
                if self.use_state_augmentation:
                    weight = torch.where(target_Qc > next_costs_state * 10, torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
                else:
                    weight = torch.where(target_Qc > self.safety_threshold, torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))

        if not self.use_safety_lagrange:
            target_q = rewards + self._gamma * (1 - terminals) * next_q * weight 
        else:
            target_q = rewards + self._gamma * (1 - terminals) * next_q 
        q1, q2 = self.critic1(obss_with_context, actions), self.critic2(obss_with_context, actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        # compute conservative loss
        random_actions = torch.FloatTensor(
            batch_size * self._num_repeat_actions, actions.shape[-1]
        ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
        # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
        tmp_obss = obss_with_context.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss_with_context.shape[-1])
        tmp_next_obss = next_obss_with_context.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, next_obss_with_context.shape[-1])
        if self.use_state_augmentation:
            old_tmp_obss = old_obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, old_obss.shape[-1])
            old_tmp_next_obss = old_next_obss.unsqueeze(1) \
                .repeat(1, self._num_repeat_actions, 1) \
                .view(batch_size * self._num_repeat_actions, old_obss.shape[-1])
        
        obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss, tmp_obss)
        next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss, tmp_obss)
        random_value1, random_value2 = self.calc_random_values(tmp_obss, random_actions)

        for value in [
            obs_pi_value1, obs_pi_value2, next_obs_pi_value1, next_obs_pi_value2,
            random_value1, random_value2
        ]:
            value = value.reshape(batch_size, self._num_repeat_actions, 1)
        
        # cat_q shape: (batch_size, 3 * num_repeat, 1)
        cat_q1 = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)
        cat_q2 = torch.cat([obs_pi_value2, next_obs_pi_value2, random_value2], 1)

        conservative_loss1 = \
            torch.logsumexp(cat_q1 / self._temperature, dim=1).mean()  * self._temperature * self._cql_weight - \
            q1.mean() * self._cql_weight
        conservative_loss2 = \
            torch.logsumexp(cat_q2 / self._temperature, dim=1).mean()  * self._temperature * self._cql_weight - \
            q2.mean() * self._cql_weight
        
        if self._with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), 0.0, 1e6)
            conservative_loss1 = cql_alpha * (conservative_loss1 - self._lagrange_threshold)
            conservative_loss2 = cql_alpha * (conservative_loss2 - self._lagrange_threshold)

            self.cql_alpha_optim.zero_grad()
            cql_alpha_loss = -(conservative_loss1 + conservative_loss2) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optim.step()
        
        if self.use_conservative_reward_loss:
            critic1_loss = critic1_loss + conservative_loss1 
            critic2_loss = critic2_loss + conservative_loss2 
        # update critic
        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # cost critic training
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss_with_context)
            target_Qc = self.critic_c_old(next_obss_with_context, next_actions)
        # for cost discount gamma is 1 
        target_Qc = costs + (1 - terminals) * target_Qc
        current_Qc = self.critic_c(obss_with_context, actions)
        critic_c_loss = ((current_Qc - target_Qc).pow(2)).mean()

        if not self.use_vae:
            obs_pi_value_c = self.calc_pi_values_c(tmp_obss, tmp_obss)
            next_obs_pi_value_c = self.calc_pi_values_c(tmp_next_obss, tmp_obss)
            random_value_c = self.calc_random_values_c(tmp_obss, random_actions)

            for value in [
                obs_pi_value_c, next_obs_pi_value_c, random_value_c
            ]:
                value = value.reshape(batch_size, self._num_repeat_actions, 1)
            
            cat_q_c = torch.cat([obs_pi_value_c, next_obs_pi_value_c, random_value_c], 1)

            conservative_loss_c = \
                torch.logsumexp(cat_q_c / self._temperature, dim=1).mean()  * self._temperature * self.cpq_weight - \
                current_Qc.mean() * self.cpq_weight
        else:
            with torch.no_grad():
                dist = self.actor(obss_with_context)
                sampled_actions = dist.sample([self._num_repeat_actions])
                sampled_actions = sampled_actions.reshape(self._num_repeat_actions * batch_size, actions.shape[-1])

                stacked_obs = obss_with_context.unsqueeze(1) \
                    .repeat(1, self._num_repeat_actions, 1) \
                    .view(batch_size * self._num_repeat_actions, obss_with_context.shape[-1])
                stacked_org_obs = obss.unsqueeze(1) \
                    .repeat(1, self._num_repeat_actions, 1) \
                    .view(batch_size * self._num_repeat_actions, obss.shape[-1])
                random_value_c = self.calc_random_values_c(stacked_obs, sampled_actions)
                qc_sampled = random_value_c.reshape(batch_size, self._num_repeat_actions)
                if self.use_state_augmentation:
                    stacked_old_obs = old_obss.unsqueeze(1) \
                        .repeat(1, self._num_repeat_actions, 1) \
                        .view(batch_size * self._num_repeat_actions, old_obss.shape[-1])
                    _, mu, logv = self.vae_list[org_task](sampled_actions, stacked_old_obs)
                else:
                    _, mu, logv = self.vae_list[org_task](sampled_actions, stacked_org_obs)
                mu = mu.reshape(batch_size, self._num_repeat_actions, -1)
                logv = logv.reshape(batch_size, self._num_repeat_actions, -1)
                kld = -0.5*(1+logv-mu.pow(2)-torch.exp(logv)).sum(2)
                #quantile = torch.quantile(kld, self.kl_threshold)
                #qc_ood = ((kld >= torch.tensor(self.kl_threshold, dtype=torch.float32).to(device)).float() * qc_sampled).mean(1)
                index = (kld >= torch.tensor(self.kl_threshold, dtype=torch.float32).to(device)).float()
                index_sum = torch.clamp(index.sum(1), min=1.0)
                qc_ood = (index * qc_sampled).sum(1)/(index_sum)
            conservative_loss_c = qc_ood.mean() * self.cpq_weight
        
        if self.train_cpq_alpha:
            cpq_alpha = self.cpq_log_alpha.exp()
            conservative_loss_c = cpq_alpha * (conservative_loss_c - self._lagrange_safety_threshold)

            self.cpq_alpha_optim.zero_grad()
            cpq_alpha_loss = conservative_loss_c
            cpq_alpha_loss.backward(retain_graph=True)
            self.cpq_alpha_optim.step()
            self.cpq_log_alpha.data.clamp_(min=-5.0, max=5.0)
        
        #here is minus for we need to maximize ood cost
        origin_critic_c_loss = critic_c_loss
        if self.use_conservative_cost_loss:
            critic_c_loss = critic_c_loss - conservative_loss_c

        self.critic_c_optim.zero_grad()
        #critic_c_loss.backward(retain_graph=True)
        critic_c_loss.backward()
        self.critic_c_optim.step()

        self._sync_weight()

        result =  {
            str(task)+"/loss/actor": actor_loss.item(),
            str(task)+"/loss/critic1": critic1_loss.item(),
            str(task)+"/loss/critic2": critic2_loss.item(),
            str(task)+"/loss/critic_c": critic_c_loss.item(),
            str(task)+"/loss/origin_critic_c": origin_critic_c_loss.item(),
            str(task)+"/loss/conservative_c": conservative_loss_c.item(),
            str(task)+"/cost_critic/cost_batch": current_Qc.mean().item(),
            str(task)+"/cost_critic/cost_batch_max": current_Qc.max().item()
        }
        if self.use_safety_lagrange:
            result[str(task)+"/cost_critic/cost_pi"] = qca.mean().item()
            result[str(task)+"/cost_critic/cost_pi_max"] = qca.max().item()
            result[str(task)+"/reward_critic/reward_pi_1"] = q1a.mean().item()
            result[str(task)+"/reward_critic/reward_pi_2"] = q2a.mean().item()

        if self._is_auto_alpha and self.policy_train=="sac":
            result[str(task)+"/loss/alpha"] = alpha_loss.item()
            result[str(task)+"/alpha"] = self._alpha.item()
        if self._with_lagrange:
            result[str(task)+"/loss/cql_alpha"] = cql_alpha_loss.item()
            result[str(task)+"/cql_alpha"] = cql_alpha.item()
        if self.train_cpq_alpha:
            result[str(task)+"/loss/cpq_alpha"] = cpq_alpha_loss.item()
            result[str(task)+"/cpq_alpha"] = cpq_alpha.item()
        if self.use_safety_lagrange:
            if self.lagrange_type == "common":
                result[str(task)+"/safety_lagrange"] = self.lagrange[task].lagrangian_multiplier.item()
            elif self.lagrange_type == "pid":
                result[str(task)+"/safety_lagrange"] = self.lagrange[task].lagrangian_multiplier
            else:
                raise ValueError("Not implemented yet.")
        
        return result
    
    def _sync_weight(self) -> None:
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic_c_old.parameters(), self.critic_c.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

