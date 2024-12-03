from typing import Unpack
from m2td3.trainer import Trainer  # NOTE: This is the trainer from M2TD3
from m2td3.algo import M2TD3Config
from fire import Fire
from dotenv import load_dotenv

load_dotenv()


def main(
    experiment_name: str,
    env_name: str,
    nb_uncertainty_dim: int,
    seed: int,
    output_dir: str,
    start_steps: int = 1e5,
    evaluate_qvalue_interval: int = 1e4,
    logger_interval: int = 1e5,
    evaluate_interval: int = 1e5,
    max_steps: int = 2e6,
    device: str = "cuda:0",
    oracle_parameters_agent: bool = False,
    **kwargs: Unpack[M2TD3Config],
):
    """Train an agent using the M2TD3 algorithm.

    This function sets up and trains a reinforcement learning agent using the M2TD3
    algorithm. The training environment, evaluation criteria, and algorithm configurations
    are customizable through parameters and additional configuration settings.

    Args:
        experiment_name (str): Name of the experiment for logging and tracking purposes.
        env_name (str): Name of the environment in which to train the agent.
        nb_uncertainty_dim (int): Number of uncertainty dimensions for the environment.
        seed (int): Random seed for reproducibility.
        output_dir (str): Directory where training outputs, logs, and models will be saved.
        start_steps (int, optional): Number of initial steps for random action exploration. Defaults to 1e5.
        evaluate_qvalue_interval (int, optional): Interval (in steps) at which to evaluate Q-values. Defaults to 1e4.
        logger_interval (int, optional): Interval (in steps) for logging training metrics. Defaults to 1e5.
        evaluate_interval (int, optional): Interval (in steps) for evaluating agent performance. Defaults to 1e5.
        max_steps (int, optional): Maximum number of training steps. Defaults to 2e6.
        device (str, optional): Device to use for training (e.g., "cuda:0" or "cpu"). Defaults to "cuda:0".
        oracle_parameters_agent (bool, optional): Whether the agent has access to MDP parameters. Defaults to False.
        **kwargs (Unpack[M2TD3Config]): Additional algorithm-specific configuration settings.


    """
    trainer = Trainer(
        experiment_name=experiment_name,
        env_name=env_name,
        nb_uncertainty_dim=nb_uncertainty_dim,
        device=device,
        seed=seed,
        output_dir=output_dir,
        start_steps=start_steps,
        evaluate_qvalue_interval=evaluate_qvalue_interval,
        logger_interval=logger_interval,
        evaluate_interval=evaluate_interval,
        max_steps=max_steps,
        oracle_parameters_agent=oracle_parameters_agent,
        **kwargs,
    )
    trainer.main()


if __name__ == "__main__":
    Fire(main)
