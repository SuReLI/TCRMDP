import uuid
import os
import socket
from typing import Unpack
import rrls
from td3.trainer import Trainer
from td3.td3 import TD3Config
from utils import env_factory, bound_factory
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
    project_name: str = "dr_dev",
    output_dir: str | None = None,
    **kwargs: Unpack[TD3Config],
):
    """Train a reinforcement learning model with domain randomization.

    This script trains a reinforcement learning agent using the TD3 algorithm within a
    domain-randomized environment. It supports configurable hyperparameters for training
    and tracks progress via optional logging.

    Args:
        experiment_name (str, optional): Name of the experiment. Defaults to "test".
        env_name (str, optional): Name of the training environment. Defaults to "Walker".
        nb_uncertainty_dim (int, optional): Number of uncertainty dimensions to randomize. Defaults to 3.
        max_steps (int, optional): Total number of training steps. Defaults to 1,000,000.
        start_steps (int, optional): Initial steps for random exploration before policy learning. Defaults to 25,000.
        seed (int, optional): Seed for random number generation. Defaults to 0.
        eval_freq (int, optional): Frequency (in steps) of evaluation. Defaults to 10,000.
        track (bool, optional): Whether to log and track training progress. Defaults to True.
        device (str, optional): Compute device for training (e.g., "cuda:0" or "cpu"). Defaults to "cuda:0".
        project_name (str, optional): Project name for tracking and logging. Defaults to "dr_dev".
        output_dir (str | None, optional): Directory for saving training results. If None, results are not saved. Defaults to None.
        **kwargs (Unpack[TD3Config]): Additional configuration options for the TD3 algorithm.
    """
    unique_id = str(uuid.uuid4())
    if output_dir is not None:
        os.makedirs(f"{output_dir}/{unique_id}", exist_ok=True)
    env = env_factory(env_name=env_name)
    eval_env = env_factory(env_name=env_name)
    params_bound = bound_factory(env_name=env_name, nb_dim=nb_uncertainty_dim)
    env = rrls.wrappers.DomainRandomization(env, params_bound=params_bound)

    params = {
        "env_name": env_name,
        "nb_uncertainty_dim": nb_uncertainty_dim,
        "seed": seed,
        "machine_name": socket.gethostname(),
    }
    trainer = Trainer(
        env,
        eval_env=eval_env,
        device=device,
        save_dir=output_dir,
        params=params,
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
    Fire(main)
