import time
from collections import OrderedDict

import gym
import torch

import cs285.infrastructure.pytorch_util as ptu
from cs285.critics.sac_critic import SACCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.sac_policy import MLPPolicySAC
from .base_agent import BaseAgent
from ..infrastructure import sac_utils


class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, times, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO: 
        # 1. Compute the target Q value. 
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic

        if len(times)==0:
            times.append(0)
            times.append(0)
            times.append(0)
            times.append(0)
            times.append(0)
            times.append(0)
            times.append(0)
            times.append(0)
            times.append(0)

        t0 = time.time()
        next_ob_no = ptu.from_numpy(next_ob_no)
        terminal_n = ptu.from_numpy(terminal_n)
        re_n = ptu.from_numpy(re_n)
        ac_na = ptu.from_numpy(ac_na)
        ob_no = ptu.from_numpy(ob_no)
        times[0] += time.time() - t0

        t1 = time.time()
        dist = self.actor(next_ob_no)
        times[1] += time.time()-t1

        t2 = time.time()
        next_action = dist.rsample()
        times[2] += time.time() - t2

        t3 = time.time()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        times[3] += time.time() - t3

        t4 = time.time()
        targets = self.critic_target(next_ob_no, next_action)
        times[4] += time.time() - t4

        t5 = time.time()
        min_q_t_pred = torch.min(*targets)
        y_target = min_q_t_pred - self.actor.alpha.detach() * log_prob

        y = re_n + self.gamma * (1 - terminal_n) * y_target
        times[5] += time.time() - t5

        t6 = time.time()
        q1, q2 = self.critic(ob_no, ac_na)
        times[6] += time.time() - t6

        t7 = time.time()
        critic_loss = self.critic.loss(q1, y.detach()) + self.critic.loss(q2, y.detach())
        times[7] += time.time() - t7

        t8 = time.time()
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        times[8] += time.time() - t8

        return critic_loss.item()

    def train(self, times, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        if len(times) == 0:
            times.append(0)
            times.append(0)
            times.append(0)
            times.append([])

        # row vector to column vector
        re_n = np.array(re_n).reshape(-1, 1)
        terminal_n = np.array(terminal_n).reshape(-1, 1)



        self.training_step += 1
        # TODO 
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)

        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor
        t0 = time.time()
        critic_loss = 0.0
        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.update_critic(times[3],ob_no, ac_na, next_ob_no, re_n, terminal_n)
        times[0] += time.time() - t0

        t1 = time.time()
        if self.training_step % self.critic_target_update_frequency == 0:
            sac_utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
        times[1] += time.time() - t1

        t2 = time.time()
        actor_loss, alpha_loss, alpha = 0.0, 0.0, 0.0
        if self.training_step % self.actor_update_frequency == 0:
            for i in range(self.agent_params['num_actor_updates_per_agent_update']):
                actor_loss, alpha_loss, alpha = self.actor.update(ob_no, self.critic)
        times[2] += time.time() - t2

        # 4. gather losses for logging
        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['Alpha_Loss'] = alpha_loss
        loss['Temperature'] = alpha



        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
