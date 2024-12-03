import os
import uuid
import socket
import gymnasium as gym
import rrls
from td3.trainer import TrainerAdversarial
from utils import env_factory, bound_factory
from tc_mdp import EvalOracleTCMDP, OracleTCMDP
from fire import Fire
from dotenv import load_dotenv

load_dotenv()


def main(
    env_name: str = "Walker",
    project_name: str = "dev_oracle_tc_rarl_001",
    nb_uncertainty_dim: int = 3,
    max_steps: int = 50_000,
    start_steps: int = 25_000,
    seed: int = 0,
    eval_freq: int = 10_000,
    track: bool = True,
    radius: float = 0.001,
    omniscient_adversary: bool = True,
    device: str = "cuda:0",
    output_dir: str = "results",
    **kwargs,
):
    """Train an agent using Oracle Time-Constrained RARL.

    This function sets up and trains an agent using the Oracle Time-Constrained
    Robust Adversarial Reinforcement Learning (RARL) algorithm. The environment is
    enhanced with time constraints and adversarial elements to test robustness.

    Args:
        env_name (str, optional): Name of the environment for training. Defaults to "Walker".
        project_name (str, optional): Name of the project for tracking. Defaults to "dev_oracle_tc_rarl_001".
        nb_uncertainty_dim (int, optional): Number of uncertainty dimensions for the environment. Defaults to 3.
        max_steps (int, optional): Maximum number of training steps. Defaults to 50,000.
        start_steps (int, optional): Number of random action steps before policy learning begins. Defaults to 25,000.
        seed (int, optional): Seed for random number generation. Defaults to 0.
        eval_freq (int, optional): Frequency (in steps) of agent evaluation. Defaults to 10,000.
        track (bool, optional): Whether to enable training progress tracking. Defaults to True.
        radius (float, optional): Radius for uncertainty bound in the OracleTCMDP. Defaults to 0.001.
        omniscient_adversary (bool, optional): Whether the adversary can observe the agent's actions. Defaults to True.
        device (str, optional): Device to use for training computations (e.g., "cuda:0" or "cpu"). Defaults to "cuda:0".
        output_dir (str, optional): Directory to save results, logs, and models. Defaults to "results".
        **kwargs: Additional parameters for `TrainerAdversarial`.
    """

    radius_str = str(radius).replace(".", "_")
    unique_id = str(uuid.uuid4())
    if output_dir is not None:
        os.makedirs(f"{output_dir}/{unique_id}", exist_ok=True)

    env = env_factory(env_name)
    eval_env = env_factory(env_name)
    params_bound = bound_factory(env_name, nb_uncertainty_dim)
    env = OracleTCMDP(
        env,
        params_bound=params_bound,
        radius=radius,
        omniscient_adversary=omniscient_adversary,
    )
    eval_env = EvalOracleTCMDP(
        eval_env,
        params_bound=params_bound,
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
    trainer = TrainerAdversarial(
        env=env,
        eval_env=eval_env,
        device=device,
        params=params,
        omniscient_adversary=omniscient_adversary,
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
