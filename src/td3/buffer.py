import torch
from typing import NamedTuple, Unpack
import random


class Transition(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    complementary_info: dict[str, torch.Tensor] = None


class BatchTransition(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor


class BatchAndPsiTransition(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor
    psi: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, device: str = "cuda:0"):
        self.capacity = capacity
        self.device = device
        self.memory: list[Transition] = []
        self.position = 0

    def add(self, *args, **kwargs: Unpack[Transition]):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args, **kwargs)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> BatchTransition:
        list_of_transitions = random.sample(self.memory, batch_size)
        batch_obs = torch.stack([t.state for t in list_of_transitions]).to(self.device)
        batch_actions = torch.stack([t.action for t in list_of_transitions]).to(
            self.device
        )
        batch_rewards = torch.tensor(
            [t.reward for t in list_of_transitions], dtype=torch.float32
        ).to(self.device)
        batch_next_obs = torch.stack([t.next_state for t in list_of_transitions]).to(
            self.device
        )
        batch_dones = torch.tensor(
            [t.done for t in list_of_transitions], dtype=torch.float32
        ).to(self.device)

        return BatchTransition(
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
        )

    def sample_batch_and_psi(self, batch_size: int) -> BatchAndPsiTransition:
        list_of_transitions = random.sample(self.memory, batch_size)
        batch_obs = torch.stack([t.state for t in list_of_transitions]).to(self.device)
        batch_actions = torch.stack([t.action for t in list_of_transitions]).to(
            self.device
        )
        batch_rewards = torch.tensor(
            [t.reward for t in list_of_transitions], dtype=torch.float32
        ).to(self.device)
        batch_next_obs = torch.stack([t.next_state for t in list_of_transitions]).to(
            self.device
        )
        batch_dones = torch.tensor(
            [t.done for t in list_of_transitions], dtype=torch.float32
        ).to(self.device)
        batch_psi = torch.stack(
            [t.complementary_info["psi"] for t in list_of_transitions]
        ).to(self.device)

        return BatchAndPsiTransition(
            batch_obs,
            batch_actions,
            batch_rewards,
            batch_next_obs,
            batch_dones,
            batch_psi,
        )
