import copy
from typing import TypedDict

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from td3.buffer import BatchAndPsiTransition, BatchTransition

from .buffer import ReplayBuffer
from .models import Actor, QNetwork, PerceptualModel


class TD3Config(TypedDict):
    buffer_size: int
    learning_rate: float
    gamma: float
    tau: float
    policy_noise: float
    exploration_noise: float
    learning_starts: int
    policy_frequency: int
    batch_size: int
    noise_clip: float
    action_min: float
    action_max: float
    device: str


class TD3Agent:
    def __init__(
        self,
        actor: Actor,
        qf1: QNetwork,
        qf2: QNetwork,
        buffer_size: int = int(1e6),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        exploration_noise: float = 0.1,
        learning_starts: int = 25e3,
        policy_frequency: int = 2,
        batch_size: int = 256,
        noise_clip: float = 0.5,
        action_min: float = -1,
        action_max: float = 1,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self.actor = actor
        # TODO see if we really need a target actor
        self.target_actor = copy.deepcopy(actor)
        self.qf1 = qf1
        self.qf2 = qf2
        self.qf1_target = copy.deepcopy(qf1)
        self.qf2_target = copy.deepcopy(qf2)
        self.q_optimizer = optim.Adam(
            list(qf1.parameters()) + list(qf2.parameters()), lr=learning_rate
        )
        self.actor_optimizer = optim.Adam(list(actor.parameters()), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, device=device)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.exploration_noise = exploration_noise
        self.learning_starts = learning_starts
        self.policy_frequency = policy_frequency
        self.batch_size = batch_size
        self.noise_clip = noise_clip
        self.action_min = action_min
        self.action_max = action_max
        self.device = device

    def select_action(self, obs: np.ndarray, use_random: bool = False) -> np.ndarray:
        if use_random:
            action = np.random.uniform(
                self.action_min, self.action_max, self.actor.action_dim
            )
        else:
            with torch.no_grad():
                action = self.actor(torch.Tensor(obs).to(self.device))
                action += torch.normal(
                    mean=0, std=self.actor.action_scale * self.exploration_noise
                )
                action = action.cpu().numpy().clip(self.action_min, self.action_max)
        return action

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: dict[str, float] = None,
    ) -> None:
        obs = torch.Tensor(obs).to(self.device)
        next_obs = torch.Tensor(next_obs).to(self.device)
        action = torch.Tensor(action).to(self.device)
        self.replay_buffer.add(obs, action, reward, next_obs, done, info)

    def update(self, global_step: int) -> dict[str, float]:
        transition = self._get_transition()
        qf1_a_values, qf2_a_values, qf1_loss, qf2_loss, qf_loss = self._update_critics(
            transition
        )
        losses = {
            "qf1_value": qf1_a_values.mean().item(),
            "qf2_value": qf2_a_values.mean().item(),
            "qf1_loss": qf1_loss.item(),
            "qf2_loss": qf2_loss.item(),
            "qf_loss": qf_loss.item(),
        }

        if global_step % self.policy_frequency == 0:
            actor_loss = self._update_actor(transition)
            losses["actor_loss"] = actor_loss.item()
            # update the target network
            self._update_target_networks()

        return losses

    def _get_transition(self):
        return self.replay_buffer.sample(self.batch_size)

    def _update_actor(self, transition):
        critic_obs = self._get_critic_obs(transition)
        actor_obs = self._get_actor_obs(transition)
        actor_loss = -self.qf1(critic_obs, self.actor(actor_obs)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def _get_critic_obs(self, transition):
        return transition.state

    def _get_critic_next_obs(self, transition):
        return transition.next_state

    def _get_actor_obs(self, transition):
        return transition.state

    def _get_actor_next_obs(self, transition):
        return transition.next_state

    def _update_target_networks(self):
        for param, target_param in zip(
            self.actor.parameters(), self.target_actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(
            self.qf1.parameters(), self.qf1_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(
            self.qf2.parameters(), self.qf2_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def _update_critics(self, transition):
        critic_obs = self._get_critic_obs(transition)
        critic_next_obs = self._get_critic_next_obs(transition)
        actor_next_obs = self._get_actor_next_obs(transition)
        with torch.no_grad():
            clipped_noise = (
                torch.randn_like(transition.action, device=self.device)
                * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip) * self.target_actor.action_scale

            next_state_action = (
                self.target_actor(actor_next_obs) + clipped_noise
            ).clamp(self.action_min, self.action_max)
            qf1_next_target = self.qf1_target(critic_next_obs, next_state_action)
            qf2_next_target = self.qf2_target(critic_next_obs, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = transition.reward.flatten() + (
                1 - transition.done.flatten() * 1
            ) * self.gamma * (min_qf_next_target).view(-1)

        qf1_a_values = self.qf1(critic_obs, transition.action).view(-1)
        qf2_a_values = self.qf2(critic_obs, transition.action).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()
        return qf1_a_values, qf2_a_values, qf1_loss, qf2_loss, qf_loss

    def save(self, path: str) -> None:
        checkpoint = {
            "actor": self.actor.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "qf1_target": self.qf1_target.state_dict(),
            "qf2_target": self.qf2_target.state_dict(),
        }
        torch.save(checkpoint, path)


class TD3OmniscientCritic(TD3Agent):
    def _get_transition(self) -> BatchAndPsiTransition:
        return self.replay_buffer.sample_batch_and_psi(self.batch_size)

    def _get_critic_obs(self, transition: BatchAndPsiTransition) -> torch.Tensor:
        return torch.cat([transition.state, transition.psi], dim=1)

    def _get_critic_next_obs(self, transition: BatchAndPsiTransition) -> torch.Tensor:
        return torch.cat([transition.next_state, transition.psi], dim=1)

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        complementary_info: dict[str, float] = None,
    ) -> None:
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        complementary_info = {
            "psi": torch.tensor(
                list(complementary_info["psi"].values()), dtype=torch.float32
            ).to(self.device),
        }
        self.replay_buffer.add(obs, action, reward, next_obs, done, complementary_info)


class M2TD3Adversary(TD3OmniscientCritic):
    # TODO Many args here are not used, should we remove them?
    """
    This

    """

    def __init__(
        self,
        actor: Actor,
        qf1: QNetwork,
        qf2: QNetwork,
        replay_buffer: ReplayBuffer,
        buffer_size: int = int(1e6),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        exploration_noise: float = 0.1,
        learning_starts: int = 25e3,
        policy_frequency: int = 2,
        batch_size: int = 256,
        noise_clip: float = 0.5,
        action_min: float = -1,
        action_max: float = 1,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self.actor = actor
        self.qf1 = qf1
        self.qf2 = qf2
        self.actor_optimizer = optim.Adam(list(actor.parameters()), lr=learning_rate)
        self.replay_buffer = replay_buffer
        self.exploration_noise = exploration_noise
        self.learning_starts = learning_starts
        self.policy_frequency = policy_frequency
        self.batch_size = batch_size
        self.action_min = action_min
        self.action_max = action_max
        self.device = device

    def _update_actor(self, transition):
        # adversarial input is [obs , psi , action]
        adv_input = torch.cat(
            [transition.state, transition.psi, transition.action], dim=1
        )
        adv_psi = self.adversary(adv_input)
        # the critic obs input is the concatenation of the state and the psi (here an adversarial one), it will be completed by the action
        critic_obs_input = torch.cat([transition.state, adv_psi], dim=1)
        adversary_loss = self.qf1(
            critic_obs_input, transition.action
        ).mean()  # here we want to minimize Q instead of maximizing it
        self.actor_optimizer.zero_grad()
        adversary_loss.backward()
        self.actor_optimizer.step()
        return adversary_loss

    def update(self, global_step: int) -> dict[str, float]:
        transition = self.replay_buffer.sample(self.batch_size)
        if global_step % self.policy_frequency == 0:
            actor_loss = self._update_actor(transition)
        return {"actor_loss": actor_loss.item()}

    def save(self, path: str) -> None:
        checkpoint = {
            "adversary": self.actor.state_dict(),
            "target_adversary": self.target_actor.state_dict(),
        }
        torch.save(checkpoint, path)
