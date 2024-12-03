import os
import uuid
import socket
import rrls
from td3.trainer import TrainerM2TD3
from utils import env_factory, bound_factory
from tc_mdp import EvalOracleTCMDP, OracleTCMDP
from fire import Fire
from dotenv import load_dotenv

load_dotenv()


def main(
    env_name: str = "Walker",
    project_name: str = "dev_oracle_tc_m2td3_001",
    nb_uncertainty_dim: int = 3,
    max_steps: int = 50_000,
    start_steps: int = 25_000,
    seed: int = 0,
    eval_freq: int = 10_000,
    track: bool = True,
    radius: float = 0.001,
    device: str = "cuda:0",
    output_dir: str = "result",
    **kwargs,
):
    """Train an agent using Oracle Time-Constrained M2TD3.

    This function sets up and trains an agent using the Oracle Time-Constrained M2TD3
    algorithm in a domain-randomized environment. It includes support for uncertainty
    modeling, time-constrained MDPs, and adversarial setups.

    Args:
        env_name (str, optional): Name of the environment to train in. Defaults to "Walker".
        project_name (str, optional): Name of the project for logging and tracking. Defaults to "dev_oracle_tc_m2td3_001".
        nb_uncertainty_dim (int, optional): Number of uncertainty dimensions for the environment. Defaults to 3.
        max_steps (int, optional): Maximum number of training steps. Defaults to 50,000.
        start_steps (int, optional): Number of steps for random action exploration at the start. Defaults to 25,000.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        eval_freq (int, optional): Frequency (in steps) for agent evaluation. Defaults to 10,000.
        track (bool, optional): Whether to enable tracking of training progress. Defaults to True.
        radius (float, optional): Radius for the uncertainty bound in the OracleTCMDP. Defaults to 0.001.
        device (str, optional): Compute device for training (e.g., "cuda:0" or "cpu"). Defaults to "cuda:0".
        output_dir (str, optional): Directory where training outputs and models will be saved. Defaults to "result".
        **kwargs: Additional parameters for `TrainerM2TD3`.
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
        omniscient_adversary=True,
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
        "omniscient_adversary": True,
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
