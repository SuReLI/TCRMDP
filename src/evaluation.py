""" """

import gymnasium as gym
from typing import Protocol
import numpy as np


class Scheduler(Protocol):
    def sample(self, t: int) -> dict[str, float]: ...


class Agent(Protocol):
    def select_action(
        self, obs: np.ndarray, use_random: bool = False
    ) -> np.ndarray: ...


def evaluate(
    env: gym.Env,
    agent: Agent,
    seed: int,
    num_episodes: int = 10,
    use_random: bool = False,
) -> list[float]:
    """
    Evaluate the performance of an agent in an environment over multiple episodes.

    Args:
        env (gym.Env): The environment to evaluate the agent in.
        agent (Agent): The agent to evaluate.
        seed (int): The seed value for the environment's random number generator.
        num_episodes (int, optional): The number of episodes to run. Defaults to 10.
        use_random (bool, optional): Whether to use random actions instead of the agent's policy. Defaults to False.

    Returns:
        list[float]: A list of episode rewards obtained by the agent.
    """
    rewards = []
    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        truncated = False
        episode_reward = 0
        while not (done or truncated):
            action = agent.select_action(obs=obs, use_random=use_random)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return rewards


def evaluate_adversarial(
    env: gym.Env,
    agent: Agent,
    adversary: Agent,
    seed: int,
    num_episodes: int = 10,
    omniscient_adversary: bool = False,
) -> list[float]:
    """
    Evaluate the performance of an agent against an adversary in an environment.

    Args:
        env (gym.Env): The environment to evaluate the agents in.
        agent (Agent): The agent being evaluated.
        adversary (Agent): The adversary agent.
        seed (int): The seed value for environment reset.
        num_episodes (int, optional): The number of episodes to evaluate. Defaults to 10.

    Returns:
        list[float]: A list of episode rewards obtained by the agent.
    """
    rewards = []
    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        truncated = False
        episode_reward = 0
        while not (done or truncated):
            action = agent.select_action(obs[0], use_random=False)
            adversary_observation = obs[1]
            if omniscient_adversary:
                adversary_observation = np.concatenate([adversary_observation, action])
            adv_action = adversary.select_action(
                adversary_observation, use_random=False
            )
            obs, reward, done, truncated, _ = env.step((action, adv_action))
            episode_reward += reward
        rewards.append(episode_reward)
    return rewards


def evaluate_with_scheduler(
    env: gym.Env, agent: Agent, scheduler: Scheduler, seed: int, num_episodes: int = 10
) -> list[float]:
    """
    Evaluates the performance of an agent in a given environment where the environment's parameters are scheduled.

    Args:
        env (gym.Env): The environment to evaluate the agent in.
        agent (Agent): The agent to evaluate.
        scheduler (Scheduler): The scheduler to use during evaluation.
        seed (int): The seed value for the environment.
        num_episodes (int, optional): The number of episodes to run. Defaults to 10.

    Returns:
        list[float]: A list of episode rewards obtained during evaluation.
    """
    rewards = []
    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        truncated = False
        episode_reward = 0
        t = 0
        scheduler.reset(seed=seed + i)
        while not (done or truncated):
            new_params = scheduler.sample(t)
            env.set_params(**new_params)
            action = agent.select_action(obs, use_random=False)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            t += 1
        rewards.append(episode_reward)
    return rewards
