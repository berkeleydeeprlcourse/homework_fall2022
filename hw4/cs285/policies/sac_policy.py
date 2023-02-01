import time

from cs285.infrastructure.sac_utils import SquashedNormal
from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools


class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20, 2],
                 action_range=[-1, 1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        entropy = self.log_alpha.exp()
        return entropy

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution

        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        # with torch.no_grad():
        observation = ptu.from_numpy(observation)
        distribution = self(observation)

        if sample:
            action = distribution.sample()
        else:
            action = distribution.mean

        action = action.clamp(*self.action_range)
        # assert action.ndim == 2 and action.shape[0] == 1

        # action = action[0]
        action = ptu.to_numpy(action)
        return action

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file

        if self.discrete:
            raise Exception("Discrete action space not implemented yet!")

        batch_mean = self.mean_net(observation)
        batch_dim = batch_mean.shape[0]

        # clip log values
        log_std = torch.tanh(self.logstd)
        log_std = log_std.clamp(*self.log_std_bounds)
        std = torch.exp(log_std)
        # scale_tril = torch.diag(std)
        # batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)

        # SquashedNormal from sac_utils file
        action_distribution = SquashedNormal(
            batch_mean,
            std,
        )

        return action_distribution

    def update(self, times, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value

        if len(times) == 0:
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
        obs = ptu.from_numpy(obs)
        times[0] += time.time() - t0

        t1 = time.time()
        dist = self(obs)
        times[1] += time.time() - t1

        t2 = time.time()
        a_tilda = dist.rsample()
        times[2] += time.time() - t2

        t3 = time.time()
        log_prob = dist.log_prob(a_tilda).sum(-1, keepdim=True)
        times[3] += time.time() - t3

        t4 = time.time()
        critic_pred = critic(obs, a_tilda)
        times[4] += time.time() - t4

        t5 = time.time()
        actor_loss = -(torch.min(*critic_pred) - self.alpha.detach() * log_prob).mean()
        times[5] += time.time() - t5

        t6 = time.time()
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        times[6] += time.time() - t6

        t7 = time.time()
        alpha_loss = -(self.alpha * (log_prob + self.target_entropy).detach()).mean()
        times[7] += time.time() - t7

        t8 = time.time()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        times[8] += time.time() - t8

        actor_loss = actor_loss.item()
        alpha_loss = alpha_loss.item()

        return actor_loss, alpha_loss, self.alpha
