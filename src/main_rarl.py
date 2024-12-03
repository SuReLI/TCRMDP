import os
import rrls
from typing import Dict, List, Unpack
import uuid
from td3.trainer import TrainerAdversarial
from td3.td3 import TD3Config
from utils import env_factory, bound_factory, DuplicateObservation, ParametersObservable
from fire import Fire
from dotenv import load_dotenv

load_dotenv()


def main(
    experiment_name: str = "test",
    env_name: str = "Walker",
    nb_uncertainty_dim: int = 3,
    max_steps: int = 1_000_000,
    start_steps: int = 25_000,
    seed: int = 0,
    eval_freq: int = 10_000,
    track: bool = True,
    device: str = "cuda:0",
    project_name: str = "rarl_dev",
    output_dir: str | None = None,
    omniscient_adversary: bool = False,
    oracle_parameters_agent: bool = False,
    **kwargs: Unpack[TD3Config],
):
    """Train an adversarial reinforcement learning model using RARL.

    This function sets up and trains an adversarial reinforcement learning model using
    the Robust Adversarial Reinforcement Learning (RARL) framework. It handles environment
    initialization, adversarial setup, and training via the TD3 algorithm with configurable
    parameters.

    Args:
        experiment_name (str, optional): Name of the experiment. Defaults to "test".
        env_name (str, optional): Name of the environment to train in. Defaults to "Walker".
        nb_uncertainty_dim (int, optional): Number of dimensions for uncertainty parameters. Defaults to 3.
        max_steps (int, optional): Maximum number of training steps. Defaults to 1,000,000.
        start_steps (int, optional): Number of initial random steps for exploration. Defaults to 25,000.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        eval_freq (int, optional): Frequency (in steps) of evaluation. Defaults to 10,000.
        track (bool, optional): Whether to log and track training metrics. Defaults to True.
        device (str, optional): Device to use for training (e.g., "cuda:0" or "cpu"). Defaults to "cuda:0".
        project_name (str, optional): Name of the project for tracking experiments. Defaults to "rarl_dev".
        output_dir (str | None, optional): Directory for saving training results. Defaults to None.
        omniscient_adversary (bool, optional): Whether the adversary has access to the agent's actions. Defaults to False.
        oracle_parameters_agent (bool, optional): Whether the agent has access to environment MDP parameters. Defaults to False.
        **kwargs (Unpack[TD3Config]): Additional TD3 algorithm-specific configurations.
    """
    unique_id = str(object=uuid.uuid4())
    if output_dir is not None:
        os.makedirs(name=f"{output_dir}/{unique_id}", exist_ok=True)

    env = env_factory(env_name=env_name)
    eval_env = env_factory(env_name=env_name)
    params_bound: Dict[str, List[float]] = bound_factory(
        env_name=env_name, nb_dim=nb_uncertainty_dim
    )
    params = {
        "env_name": env_name,
        "nb_uncertainty_dim": nb_uncertainty_dim,
        "seed": seed,
        "omniscient_adversary": omniscient_adversary,
    }
    if oracle_parameters_agent:
        env = ParametersObservable(env=env, params_bound=params_bound)
        eval_env = ParametersObservable(env=eval_env, params_bound=params_bound)
    env = rrls.wrappers.DynamicAdversarial(env, params_bound=params_bound)
    env = DuplicateObservation(env=env, omniscient_adversary=omniscient_adversary)

    trainer = TrainerAdversarial(
        env=env,
        eval_env=eval_env,
        device=device,
        save_dir=output_dir,
        params=params,
        omniscient_adversary=omniscient_adversary,
        **kwargs,
    )
    trainer.train(
        experiment_name=experiment_name,
        max_steps=max_steps,
        start_steps=start_steps,
        seed=seed,
        eval_freq=eval_freq,
        track=track,
        project_name=project_name,
    )


if __name__ == "__main__":
    Fire(component=main)
