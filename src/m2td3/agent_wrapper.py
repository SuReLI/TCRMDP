import torch
from typing import Unpack, TypedDict
from m2td3.utils import PolicyNetwork

import numpy as np


class M2TD3NetworksKwargs(TypedDict):
    state_dim: int
    action_dim: int
    max_action: int
    device: torch.device
    hidden_num: int = 1
    hidden_layer: int = 256


class M2TD3AgentWrapper:
    """Convert from the m2td3 format to a more general format."""

    def __init__(
        self,
        policy_path: str,
        seed: int = 0,
        policy_std: float = 0.1,
        **kwargs: Unpack[M2TD3NetworksKwargs],
    ) -> None:
        self.policy = PolicyNetwork(**kwargs)
        self.policy.load_state_dict(state_dict=torch.load(policy_path))
        self.device: torch.device = kwargs["device"]
        self.policy.to(device=self.device)
        self.state_dim: int = kwargs["state_dim"]
        self.rand_state = np.random.RandomState(seed=seed)
        self.policy_std: float = policy_std

    def select_action(self, obs: np.ndarray, use_random: bool = False) -> np.ndarray:
        state_tensor = torch.tensor(
            data=obs, dtype=torch.float, device=self.device
        ).view(-1, self.state_dim)
        action = self.policy(state_tensor)
        if not use_random:
            noise = torch.tensor(
                data=self.rand_state.normal(loc=0, scale=self.policy_std),
                dtype=torch.float,
                device=self.device,
            )
            action = action + noise
        return action.squeeze(0).detach().cpu().numpy()
