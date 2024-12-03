import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(
            observation_dim + action_dim,
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    # we need action_space to get min and max values
    def __init__(self, observation_dim, action_space) -> None:
        super().__init__()
        self.action_dim = np.prod(action_space.shape)
        self.fc1 = nn.Linear(observation_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, self.action_dim)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_space.high - action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_space.high + action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class PerceptualModel(nn.Module):
    def __init__(self, observation_dim, action_dim, psi_dim):
        super().__init__()
        input_size = observation_dim * 2 + action_dim
        self.fully_connected = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, psi_dim),
        )

    def forward(self, s_t, a, s_tp1):
        x = torch.cat([s_t, a, s_tp1], dim=1)
        return self.fully_connected(x)
