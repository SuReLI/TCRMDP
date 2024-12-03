from typing import Any, Optional, Unpack
import os
from collections import deque
import numpy as np
import torch
import gymnasium as gym
import rrls
from tqdm import tqdm
import wandb

from tc_mdp import TCMDP

from .models import Actor, QNetwork
from .td3 import M2TD3Adversary, TD3Agent, TD3OmniscientCritic, TD3Config


class Trainer:
    """
    Trainer for TD3 agents.

    Args:
        env (gym.Env): The environment to train on.
        device (torch.device, optional): The device to use for training. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").
        eval_env (gym.Env, optional): The environment to use for evaluation. Defaults to None.
        **kwargs: Additional arguments for the TD3 agent.
    """

    def __init__(
        self,
        env: gym.Env,
        device: torch.device = torch.device(
            device="cuda" if torch.cuda.is_available() else "cpu"
        ),
        eval_env: gym.Env = None,
        save_dir: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
        supplementary_artifacts: dict[str, str] | None = None,
        **kwargs: Unpack[TD3Config],
    ) -> None:
        self.env = env
        if eval_env is None:
            eval_env = env
        self.eval_env = eval_env
        self.device = device
        self._init_agent(env=env, device=device, kwargs=kwargs)
        if save_dir is not None:
            self.save_dir: str | None = save_dir
        else:
            self.save_dir = "tmp"
        os.makedirs(self.save_dir, exist_ok=True)
        self.params: dict[str, Any] = params if params is not None else {}
        self.supplementary_artifacts: dict[str, str] | None = supplementary_artifacts

    def _init_agent(self, env: gym.Env, device: torch.device, kwargs: dict[str, Any]):
        """
        Initialize a TD3 agent.

        Args:
            env (gym.Env): The environment to train on.
            device (torch.device): The device to use for training.
            kwargs (dict[str, Any]): Additional arguments for the TD3 agent.
        """
        action_space = env.action_space
        obs_space = env.observation_space

        observation_dim: int = np.prod(obs_space.shape)
        action_dim: int = np.prod(action_space.shape)

        actor = Actor(observation_dim=observation_dim, action_space=action_space).to(
            device
        )
        qf1 = QNetwork(observation_dim=observation_dim, action_dim=action_dim).to(
            device
        )
        qf2 = QNetwork(observation_dim=observation_dim, action_dim=action_dim).to(
            device
        )
        self.agent = TD3Agent(actor, qf1, qf2, device=device, **kwargs)

    def _store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        done: bool,
        info: dict[str, Any] = None,
    ) -> None:
        """
        Store a transition in the agent's replay buffer.

        Args:
            obs (np.ndarray): The current observation.
            action (np.ndarray): The action taken.
            next_obs (np.ndarray): The next observation.
            reward (float): The reward received.
            done (bool): Whether the episode is done.
            info (dict[str, Any]): Additional information.
        """

        self.agent.store_transition(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            info=info,
        )

    def _get_action(self, obs: np.ndarray, use_random: bool) -> np.ndarray:
        """
        Get an action from the agent.

        Args:
            obs (np.ndarray): The current observation.
            use_random (bool): Whether to use random actions.

        Returns:
            np.ndarray: The action to take.
        """
        return self.agent.select_action(obs, use_random=use_random)

    def _update_agents(self, global_step: int) -> dict[str, float]:
        """
        Make an update step for the agent.

        Args:
            global_step (int): The current global step.

        Returns:
            dict[str, float]: The losses computed during the update step.
        """

        losses = self.agent.update(global_step)
        return losses

    def _save_agent(self) -> None:
        """
        Save the trained agent to a file.
        Args:
            path (str): The path to save the agent to.
        """
        if self.save_dir is not None:
            # create a directory if it does not exist
            os.makedirs(self.save_dir, exist_ok=True)
            self.agent.save(f"{self.save_dir}/agent.pth")

    def _build_artifact(self) -> wandb.Artifact:
        """
        Build a wandb artifact for the trained agent.
        Returns:
            wandb.Artifact: The artifact containing the trained agent.
        """
        artifact = wandb.Artifact("models", type="model")
        if self.supplementary_artifacts is not None:
            for name, path in self.supplementary_artifacts.items():
                artifact.add_file(local_path=path, name=name)
        artifact.add_file(local_path=f"{self.save_dir}/agent.pth")
        return artifact

    def train(
        self,
        max_steps: int = 1e6,
        start_steps: int = 25e3,
        seed: int = 0,
        eval_freq: int = 1e4,
        track: bool = False,
        project_name: str = "tcmdp_dev",
        experiment_name: str | None = None,
        log_freq: int = 10000,
    ) -> None:
        """
        Train the agent.

        Args:
            max_steps (int, optional): The maximum number of training steps. Defaults to 1000000.
            start_steps (int, optional): The number of steps to collect transitions without updating the agent. Defaults to 25e3.
            seed (int, optional): The random seed for the environment. Defaults to 0.
            eval_freq (int, optional): The evaluation frequency. Defaults to 10000.
            track (bool, optional): Whether to track the training with wandb. Defaults to False.
            project_name (str, optional): The name of the wandb project. Defaults to "rrls_dev".
            experiment_name (str, optional): The name of the wandb experiment. Defaults to "rrls_dev".
            log_freq (int, optional): The logging frequency. Defaults to 10000.
        """

        if track:
            wandb.init(
                project=project_name,
                name=experiment_name,
                save_code=True,
                config=self.params,
            )
            episodes_reward = deque(maxlen=10)

        obs, _ = self.env.reset(seed=seed)
        total_reward = 0

        t = tqdm(range(max_steps))
        for global_step in t:
            # Evaluation
            if global_step % eval_freq == 0:
                mean_reward = self.evaluate(seed)
                t.set_description(f"Mean test reward: {mean_reward:.2f}")
                if track:
                    wandb.log({"mean_reward": mean_reward}, step=global_step)
                    if len(episodes_reward) == 10:
                        wandb.log(
                            {"episodes_reward": np.mean(episodes_reward)},
                            step=global_step,
                        )

            pre_training = global_step < start_steps
            action = self._get_action(obs, pre_training)
            next_obs, reward, done, truncated, info = self.env.step(action)
            self._store_transition(obs, action, next_obs, reward, done, info)
            total_reward += reward
            obs = next_obs
            if done or truncated:
                obs, _ = self.env.reset()
                self._reset_agent_memory()
                episodes_reward.append(total_reward)
                total_reward = 0

            if not pre_training:
                losses = self._update_agents(global_step)
                if track & (global_step % log_freq == 0):
                    wandb.log(losses, step=global_step)
        if self.save_dir is not None:
            self._save_agent()
        if track:
            artifact = self._build_artifact()
            wandb.log_artifact(artifact)

    # TODO remove?
    def evaluate(self, seed: int = 0):
        """
        Evaluate the agent on the evaluation environment.

        Args:
            seed (int, optional): The random seed for the environment. Defaults to 0.

        Returns:
            float: The mean episode reward.
        """
        episode_reward_list = []
        for _ in range(10):
            obs, _ = self.eval_env.reset(seed=seed)
            self._reset_agent_memory()
            episode_reward = 0
            while True:
                with torch.no_grad():
                    action = self.agent.select_action(obs, use_random=False)
                next_obs, reward, done, truncation, info = self.eval_env.step(action)
                episode_reward += reward
                if done or truncation:
                    episode_reward_list.append(episode_reward)
                    break
                obs = next_obs
        mean_episode_reward = np.mean(episode_reward_list)
        self._reset_agent_memory()
        return mean_episode_reward

    def _reset_agent_memory(self):
        """
        Reset the agent's internal memory of the previous state and action.
        note:: This is necessary for the PerceptualTD3Agent to reset the previous state and action.
        """
        # not implemented for basic TD3 agents
        pass


class TrainerAdversarial(Trainer):
    """
    Trainer for adversarial TD3 agents.

    Args:
        env (gym.Env): The environment to train on.
        device (torch.device, optional): The device to use for training. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").
        eval_env (gym.Env, optional): The environment to use for evaluation. Defaults to None.
        omniscient_adversary (bool, optional): Whether the adversary has access to the agent's action. Defaults to False.
        **kwargs: Additional arguments for the TD3 agents.
    """

    def __init__(
        self,
        env: gym.Env,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        eval_env: gym.Env = None,
        omniscient_adversary: bool = False,
        **kwargs: Unpack[TD3Config],
    ) -> None:
        super().__init__(env=env, device=device, eval_env=eval_env, **kwargs)
        self.omniscient_adversary = omniscient_adversary

    def _init_agent(self, env: gym.Env, device: torch.device, kwargs: dict[str, Any]):
        """
        Initialize two TD3 agents, one for the agent and one for the adversary.

        Args:
            env (gym.Env): The environment to train on.
            device (torch.device): The device to use for training.
            kwargs (dict[str, Any]): Additional arguments for the TD3 agents.
        """
        agent_action_space, adv_action_space = env.action_space
        agent_obs_space, adv_obs_space = env.observation_space

        agent_action_dim = np.prod(agent_action_space.shape)
        agent_obs_dim = np.prod(agent_obs_space.shape)
        adv_action_dim = np.prod(adv_action_space.shape)
        adv_obs_dim = np.prod(adv_obs_space.shape)
        agent_actor = Actor(
            observation_dim=agent_obs_dim, action_space=agent_action_space
        ).to(device)
        agent_qf1 = QNetwork(
            observation_dim=agent_obs_dim, action_dim=agent_action_dim
        ).to(device)
        agent_qf2 = QNetwork(
            observation_dim=agent_obs_dim, action_dim=agent_action_dim
        ).to(device)
        self.agent = TD3Agent(
            actor=agent_actor, qf1=agent_qf1, qf2=agent_qf2, device=device, **kwargs
        )

        adv_actor = Actor(
            observation_dim=adv_obs_dim, action_space=adv_action_space
        ).to(device)
        adv_qf1 = QNetwork(observation_dim=adv_obs_dim, action_dim=adv_action_dim).to(
            device
        )
        adv_qf2 = QNetwork(observation_dim=adv_obs_dim, action_dim=adv_action_dim).to(
            device
        )
        self.adversary = TD3Agent(
            actor=adv_actor, qf1=adv_qf1, qf2=adv_qf2, device=device, **kwargs
        )

    def _store_transition(
        self,
        obs: tuple[np.ndarray, np.ndarray],
        action: tuple[np.ndarray, np.ndarray],
        next_obs: tuple[np.ndarray, np.ndarray],
        reward: float,
        done: bool,
        info: dict[str, Any] = None,
    ) -> None:
        """
        Store a transition in the agents replay buffer.

        Args:
            obs (tuple[np.ndarray, np.ndarray]): The current observation.
            action (tuple[np.ndarray, np.ndarray]): The action taken.
            next_obs (tuple[np.ndarray, np.ndarray]): The next observation.
            reward (float): The reward received.
            done (bool): Whether the episode is done.
            info (dict[str, Any]): Additional information.
        """

        self.agent.store_transition(
            obs=obs[0],
            action=action[0],
            reward=reward,
            next_obs=next_obs[0],
            done=done,
            info=info,
        )

        obs_adversary = obs[1]
        next_obs_adversary = next_obs[1]
        if self.omniscient_adversary:
            # Adversary has access to the agent's action
            obs_adversary = np.concatenate(
                [obs_adversary, action[0]],
            )
            # Adversary has access to the agent's action, so we have to infer the next agent action
            with torch.no_grad():
                next_agent_action = self.agent.select_action(
                    next_obs[0], use_random=False
                )
            next_obs_adversary = np.concatenate(
                [next_obs_adversary, next_agent_action],
            )

        self.adversary.store_transition(
            obs=obs_adversary,
            action=action[1],
            reward=-reward,
            next_obs=next_obs_adversary,
            done=done,
        )

    def _get_action(
        self, obs: tuple[np.ndarray, np.ndarray], use_random: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get an action tuple from the agents.

        Args:
            obs (tuple[np.ndarray, np.ndarray]): The current observation.
            use_random (bool): Whether to use random actions.

        Returns:
            tuple[np.ndarray, np.ndarray]: The actions to take.
        """

        agent_action = self.agent.select_action(obs[0], use_random=use_random)
        obs_adversary = obs[1]
        if self.omniscient_adversary:
            # Adversary has access to the agent's action
            obs_adversary = np.concatenate(
                [obs_adversary, agent_action],
            )
        adv_action = self.adversary.select_action(obs_adversary, use_random=use_random)

        return agent_action, adv_action

    def _update_agents(self, global_step: int) -> dict[str, float]:
        """
        Make an update step for the agents.

        Args:
            global_step (int): The current global step.

        Returns:
            dict[str, float]: The losses computed during the update step.
        """

        agent_losses = self.agent.update(global_step)
        adv_losses = self.adversary.update(global_step)
        return {**agent_losses, **adv_losses}

    def _save_agent(self) -> None:
        """
        Save the trained agents to a file.
        Args:
            path (str): The path to save the agent to.
        """
        if self.save_dir is not None:
            # create a directory if it does not exist
            os.makedirs(self.save_dir, exist_ok=True)
            self.agent.save(f"{self.save_dir}/agent.pth")
            self.adversary.save(f"{self.save_dir}/adversary.pth")

    def _build_artifact(self) -> wandb.Artifact:
        """
        Build a wandb artifact for a trained agent and adversary.
        Returns:
            wandb.Artifact: The artifact containing the trained agent and the adversary.
        """
        artifact = wandb.Artifact("models", type="model")
        if self.supplementary_artifacts is not None:
            for name, path in self.supplementary_artifacts.items():
                artifact.add_file(local_path=path, name=name)
        artifact.add_file(f"{self.save_dir}/agent.pth")
        artifact.add_file(f"{self.save_dir}/adversary.pth")
        return artifact


class TrainerM2TD3(Trainer):
    """
    For time constrained M2TD3 agents, it's hard to extend the code from tanabe et al,
    we reimplement M2TD3 but without all \hat{\omega} and it's works well.
    """

    def __init__(
        self,
        env: gym.Env,
        device: torch.device = torch.device(
            device="cuda" if torch.cuda.is_available() else "cpu"
        ),
        eval_env: gym.Env = None,
        save_dir: str | None = None,
        params: dict[str, Any] | None = None,
        supplementary_artifacts: dict[str, str] | None = None,
        **kwargs: TD3Config,
    ) -> None:
        super().__init__(
            env, device, eval_env, save_dir, params, supplementary_artifacts, **kwargs
        )

    def _init_agent(self, env: gym.Env, device: torch.device, kwargs: dict[str, Any]):
        """
        Initialize a TD3 agent.

        Args:
            env (gym.Env): The environment to train on.
            device (torch.device): The device to use for training.
            kwargs (dict[str, Any]): Additional arguments for the TD3 agent.
        """
        action_space = env.action_space
        obs_space = env.observation_space

        actor_observation_dim: int = np.prod(obs_space[0].shape)
        critic_observation_dim: int = np.prod(obs_space[0].shape) + len(
            env.params_bound
        )  # the critic is omniscient and has access to the real psi
        adversary_observation_dim: int = np.prod(obs_space[1].shape)
        action_dim: int = np.prod(action_space[0].shape)

        actor = Actor(
            observation_dim=actor_observation_dim, action_space=action_space[0]
        ).to(device)
        qf1 = QNetwork(
            observation_dim=critic_observation_dim, action_dim=action_dim
        ).to(device)
        qf2 = QNetwork(
            observation_dim=critic_observation_dim, action_dim=action_dim
        ).to(device)
        adversary = Actor(
            observation_dim=adversary_observation_dim, action_space=action_space[1]
        ).to(device)
        self.agent = TD3OmniscientCritic(
            actor=actor, qf1=qf1, qf2=qf2, device=device, **kwargs
        )
        self.adversary = M2TD3Adversary(
            actor=adversary,
            qf1=qf1,
            qf2=qf2,
            replay_buffer=self.agent.replay_buffer,
            device=device,
            **kwargs,
        )

    def _get_action(self, obs: np.ndarray, use_random: bool) -> np.ndarray:
        agent_action = self.agent.select_action(obs[0], use_random=use_random)
        adv_obs = np.concatenate([obs[1], agent_action])
        adv_action = self.adversary.select_action(adv_obs, use_random=use_random)
        return agent_action, adv_action

    def _store_transition(
        self,
        obs: np.ndarray,
        action: tuple[np.ndarray, np.ndarray],
        next_obs: np.ndarray,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """
        Store a transition in the agent's replay buffer.

        Args:
            obs (np.ndarray): The current observation.
            action (np.ndarray): The action taken.
            next_obs (np.ndarray): The next observation.
            reward (float): The reward received.
            done (bool): Whether the episode is done.
            info (dict[str, Any]): Additional information.
        """

        self.agent.store_transition(
            obs=obs[0],
            action=action[0],
            reward=reward,
            next_obs=next_obs[0],
            done=done,
            complementary_info=info,
        )
