import os
import uuid
import socket
from typing import Dict, List, Unpack
import rrls
from td3.trainer import Trainer
from td3.td3 import TD3Config
from utils import env_factory, bound_factory, AgentInferenceBuilder
from fire import Fire
from dotenv import load_dotenv
from evaluation import Agent
from tc_mdp import TCMDPFixedAgent, EvalOracleTCMDP, EvalStackedTCMDP

load_dotenv()


def main(
    agent_path: str,
    agent_type: str = "m2td3",
    radius: float = 0.1,
    experiment_name: str = "test",
    env_name: str = "Walker",
    nb_uncertainty_dim: int = 3,
    max_steps: int = 1_000_000,
    start_steps: int = 25_000,
    seed: int = 0,
    eval_freq: int = 10_000,
    omniscient_adversary: bool = True,
    track: bool = True,
    project_name: str = "dev-tc-adversary",
    device: str = "cuda:0",
    output_dir: str | None = None,
    **kwargs: Unpack[TD3Config],
):
    """Train a time-constrained adversary against a trained agent.

    This function sets up and trains a time-constrained adversary in an environment
    where the agent has already been trained. The adversary can utilize omniscient
    information, and the environment is customizable with uncertainty dimensions
    and radius constraints.

    Args:
        agent_path (str): Path to the trained agent's weights file.
        agent_type (str, optional): Type of agent (e.g., "m2td3", "td3", or variants with "oracle" or "stacked"). Defaults to "m2td3".
        radius (float, optional): Radius for uncertainty bounds in the environment. Defaults to 0.1.
        experiment_name (str, optional): Name of the experiment for tracking and logging. Defaults to "test".
        env_name (str, optional): Name of the training environment. Defaults to "Walker".
        nb_uncertainty_dim (int, optional): Number of uncertainty dimensions for environment bounds. Defaults to 3.
        max_steps (int, optional): Maximum number of training steps for the adversary. Defaults to 1,000,000.
        start_steps (int, optional): Number of initial random steps for exploration. Defaults to 25,000.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        eval_freq (int, optional): Frequency (in steps) of adversary evaluation. Defaults to 10,000.
        omniscient_adversary (bool, optional): Whether the adversary has omniscient knowledge of the agent's actions. Defaults to True.
        track (bool, optional): Whether to enable tracking and logging of training metrics. Defaults to True.
        project_name (str, optional): Name of the project for tracking purposes. Defaults to "dev-tc-adversary".
        device (str, optional): Device to use for computations (e.g., "cuda:0" or "cpu"). Defaults to "cuda:0".
        output_dir (str | None, optional): Directory to save experiment outputs and logs. If None, results are not saved. Defaults to None.
        **kwargs (Unpack[TD3Config]): Additional configuration options for the TD3 algorithm.
    """
    unique_id = str(uuid.uuid4())
    if output_dir is not None:
        os.makedirs(f"{output_dir}/{unique_id}", exist_ok=True)

    env = env_factory(env_name=env_name)
    eval_env = env_factory(env_name=env_name)
    params_bound: Dict[str, List[float]] = bound_factory(
        env_name=env_name, nb_dim=nb_uncertainty_dim
    )
    if "oracle" in agent_type:
        env = EvalOracleTCMDP(env, params_bound)
        eval_env = EvalOracleTCMDP(eval_env, params_bound)
        if "m2td3" in agent_type:
            agent_type = "m2td3"
        else:
            agent_type = "td3"
    if "stacked" in agent_type:
        env = EvalStackedTCMDP(env, params_bound)
        eval_env = EvalStackedTCMDP(eval_env, params_bound)
        agent_type = "td3"

    agent_builder = AgentInferenceBuilder(
        env=env, nb_dim=nb_uncertainty_dim, device=device
    )
    agent: Agent = (
        agent_builder.add_actor_path(path=agent_path)
        .add_device(device)
        .add_agent_type(agent_type)
        .build()
    )
    env = TCMDPFixedAgent(
        env=env,
        agent=agent,
        params_bound=params_bound,
        is_omniscient=omniscient_adversary,
    )
    eval_env = TCMDPFixedAgent(
        env=eval_env,
        agent=agent,
        params_bound=params_bound,
        is_omniscient=omniscient_adversary,
    )
    params = {
        "env_name": env_name,
        "radius": radius,
        "omniscient_adversary": omniscient_adversary,
        "seed": seed,
        "machine_name": socket.gethostname(),
    }
    supplementary_artifacts = {
        "agent_trained.pth": agent_path,
    }

    radius_str = str(radius).replace(".", "_")
    project_name_concat = f"{project_name}_{radius_str}"
    trainer = Trainer(
        env=env,
        eval_env=eval_env,
        device=device,
        params=params,
        supplementary_artifacts=supplementary_artifacts,
        save_dir=output_dir,
        **kwargs,
    )
    trainer.train(
        experiment_name=experiment_name,
        max_steps=max_steps,
        start_steps=start_steps,
        seed=seed,
        eval_freq=eval_freq,
        track=track,
        project_name=project_name_concat,
    )


if __name__ == "__main__":
    Fire(main)
