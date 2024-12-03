import os
import socket
import uuid
from typing import Unpack
import rrls
from td3.trainer import Trainer
from td3.td3 import TD3Config
from utils import env_factory
from fire import Fire
from dotenv import load_dotenv

load_dotenv()


def main(
    experiment_name: str = "test",
    env_name: str = "Walker",
    max_steps: int = 1_000_000,
    start_steps: int = 25_000,
    seed: int = 0,
    eval_freq: int = 10_000,
    track: bool = True,
    device: str = "cuda:0",
    project_name: str = "vanilla",
    output_dir: str | None = None,
    **kwargs: Unpack[TD3Config],
):
    """Train an agent using the classical TD3 algorithm.

    This function trains a reinforcement learning agent using the Twin Delayed Deep
    Deterministic Policy Gradient (TD3) algorithm in a specified environment. It supports
    configurable hyperparameters and tracks training progress.

    Args:
        experiment_name (str, optional): Name of the experiment for logging and tracking purposes. Defaults to "test".
        env_name (str, optional): Name of the environment to train in. Defaults to "Walker".
        max_steps (int, optional): Maximum number of training steps. Defaults to 1,000,000.
        start_steps (int, optional): Number of random action steps for exploration before policy training. Defaults to 25,000.
        seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 0.
        eval_freq (int, optional): Frequency (in steps) at which the agent is evaluated. Defaults to 10,000.
        track (bool, optional): Whether to enable tracking and logging of training metrics. Defaults to True.
        device (str, optional): Device to use for training computations (e.g., "cuda:0" or "cpu"). Defaults to "cuda:0".
        project_name (str, optional): Name of the project for tracking and logging. Defaults to "vanilla".
        output_dir (str | None, optional): Directory to save training outputs and logs. If None, results are not saved. Defaults to None.
        **kwargs (Unpack[TD3Config]): Additional configuration parameters for the TD3 algorithm.
    """

    unique_id = str(uuid.uuid4())
    if output_dir is not None:
        os.makedirs(f"{output_dir}/{unique_id}", exist_ok=True)

    env = env_factory(env_name=env_name)
    eval_env = env_factory(env_name=env_name)
    params = {
        "env_name": env_name,
        "seed": seed,
        "machine_name": socket.gethostname(),
    }
    trainer = Trainer(
        env=env,
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
    Fire(component=main)
