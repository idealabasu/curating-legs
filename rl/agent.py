import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class Agent(nn.Module):
    def __init__(self, num_obs, num_actions):
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_obs, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(num_obs, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, num_actions)
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, num_actions))

    def get_value(self, x):
        return self.critic(x)

    def get_deterministic_action(self, x):
        return self.actor_mean(x)

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        prob = Normal(action_mean, action_std)
        if action is None:
            action = prob.sample()
        return (
            action,
            prob.log_prob(action).sum(-1),
            prob.entropy().sum(-1)
        )
