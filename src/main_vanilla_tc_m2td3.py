import os
import uuid
import socket
import gymnasium as gym
import rrls
from td3.trainer import TrainerM2TD3
from utils import env_factory, bound_factory
from tc_mdp import TCMDP
from fire import Fire
from dotenv import load_dotenv

load_dotenv()


def main(
    env_name: str = "Walker",
    project_name: str = "dev_vanilla_tc_m2td3_001",
    nb_uncertainty_dim: int = 3,
    max_steps: int = 50_000,
    start_steps: int = 25_000,
    seed: int = 0,
    eval_freq: int = 10_000,
    track: bool = True,
    radius: float = 0.001,
    omniscient_adversary: bool = True,
    device: str = "cuda:0",
    output_dir: str = "result",
    **kwargs,
):
    """Train an agent using the vanilla Time-Constrained M2TD3 algorithm.

    This function trains a reinforcement learning agent using the vanilla declination of
    the Time-Constrained (TC) algorithm integrated with the M2TD3 framework. The environment
    incorporates time constraints and can be customized with uncertainty dimensions, radius
    bounds, and optional omniscient adversary setups.

    Args:
        env_name (str, optional): Name of the environment to train in. Defaults to "Walker".
        project_name (str, optional): Name of the project for logging and tracking. Defaults to "dev_vanilla_tc_m2td3_001".
        nb_uncertainty_dim (int, optional): Number of uncertainty dimensions for the environment. Defaults to 3.
        max_steps (int, optional): Maximum number of training steps. Defaults to 50,000.
        start_steps (int, optional): Number of random action steps before policy learning begins. Defaults to 25,000.
        seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 0.
        eval_freq (int, optional): Frequency (in steps) of agent evaluation. Defaults to 10,000.
        track (bool, optional): Whether to enable tracking and logging of training metrics. Defaults to True.
        radius (float, optional): Radius for uncertainty bounds in the TCMDP environment. Defaults to 0.001.
        omniscient_adversary (bool, optional): Whether the adversary has omniscient knowledge of the agent's actions. Defaults to True.
        device (str, optional): Device to use for training computations (e.g., "cuda:0" or "cpu"). Defaults to "cuda:0".
        **kwargs: Additional configuration parameters for `TrainerM2TD3`.
    """
    radius_str = str(radius).replace(".", "_")
    unique_id = str(uuid.uuid4())
    if output_dir is not None:
        os.makedirs(f"{output_dir}/{unique_id}", exist_ok=True)

    env = env_factory(env_name)
    eval_env = env_factory(env_name)
    params_bound = bound_factory(env_name, nb_uncertainty_dim)
    env = TCMDP(
        env,
        params_bound=params_bound,
        radius=radius,
        omniscient_adversary=omniscient_adversary,
    )

    project_name_concat = f"{project_name}"
    experiment_name = f"{env_name}_{radius_str}_{unique_id}"

    params = {
        "env_name": env_name,
        "nb_uncertainty_dim": nb_uncertainty_dim,
        "radius": radius,
        "seed": seed,
        "omniscient_adversary": omniscient_adversary,
        "machine_name": socket.gethostname(),
    }
    trainer = TrainerM2TD3(
        env=env,
        eval_env=eval_env,
        device=device,
        params=params,
        save_dir=output_dir,
        **kwargs,
    )
    trainer.train(
        experiment_name=experiment_name,
        max_steps=max_steps,
        start_steps=start_steps,
        project_name=project_name_concat,
        seed=seed,
        eval_freq=eval_freq,
        track=track,
    )


if __name__ == "__main__":
    Fire(main)
