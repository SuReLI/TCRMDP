import glob
import os
import shutil
from typing import TypedDict, Unpack
import uuid

import numpy as np
import torch
import wandb

import gymnasium as gym
from m2td3.algo import M2TD3, M2TD3Config
from m2td3.factory import env_factory, bound_factory
from m2td3.utils import ParametersObservable

# from SOFT_M2TD3.soft_m2td3 import SoftM2TD3

AGENT_DICT = {
    "M2TD3": M2TD3,
    # "SoftM2TD3": SoftM2TD3,
}


class Trainer:
    """Initialize Trainer

    Parameters
    ----------
    config : Dict
        configs
    experiment_name : str
        experiment name

    """

    def __init__(
        self,
        experiment_name: str,
        env_name: str,
        nb_uncertainty_dim: int,
        device: str,
        seed: int,
        output_dir: str,
        start_steps: int = 1e5,
        evaluate_qvalue_interval: int = 1e4,
        logger_interval: int = 1e5,
        evaluate_interval: int = 1e6,
        max_steps: int = 2e6,
        track: bool = True,
        project_name: str = "m2td3_dev",
        experiement_name: str | None = None,
        oracle_parameters_agent: bool = False,
        **kwargs: Unpack[M2TD3Config],
    ):
        self.uncertainty_set = bound_factory(env_name, nb_uncertainty_dim)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.rand_state = np.random.RandomState(seed)

        unique_id = str(object=uuid.uuid4())
        self.output_dir = f"{output_dir}/{unique_id}"
        self.start_steps = start_steps
        self.seed = seed
        self.experiment_name = experiment_name

        self.evaluate_qvalue_interval = evaluate_qvalue_interval
        self.logger_interval = logger_interval
        self.evaluate_interval = evaluate_interval
        self.max_steps = max_steps

        cwd_dir = output_dir

        if os.path.exists(f"{cwd_dir}/{unique_id}/policies"):
            shutil.rmtree(f"{cwd_dir}/{unique_id}/policies")
        if os.path.exists(f"{cwd_dir}/{unique_id}/critics"):
            shutil.rmtree(f"{cwd_dir}/{unique_id}/critics")
        file_list = glob.glob(f"{cwd_dir}/{unique_id}/*")
        for file_path in file_list:
            if os.path.isfile(file_path):
                os.remove(file_path)

        os.makedirs(f"{self.output_dir}/policies", exist_ok=True)
        os.makedirs(f"{self.output_dir}/critics", exist_ok=True)

        # TODO do later

        # TODO seed
        env = env_factory(env_name)
        if oracle_parameters_agent:
            env = ParametersObservable(env=env, params_bound=self.uncertainty_set)
        self.env = env

        if isinstance(env.observation_space, gym.spaces.Dict):
            self.state_dim = len(env.observation_space.spaces)
        else:
            self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]

        self.change_param_min = np.array([v[0] for v in self.uncertainty_set.values()])
        self.change_param_max = np.array([v[1] for v in self.uncertainty_set.values()])
        # Hadle M2TD3 only for the moment
        # TODO: Handle SoftM2TD3
        self.agent = M2TD3(
            # config=config,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            omega_dim=nb_uncertainty_dim,
            max_action=self.max_action,
            rand_state=self.rand_state,
            device=self.device,
            min_omega=self.change_param_min,
            max_omega=self.change_param_max,  # TODO to confirm
            **kwargs,
        )

        self.step = 0
        self.episode_len = env.get_wrapper_attr("_max_episode_steps")

        if track:
            wandb.init(
                project=project_name,
                name=experiment_name,
                save_code=True,
            )

    def save_model(self):
        """Save networks"""
        torch.save(
            self.agent.policy_network.to("cpu").state_dict(),
            f"{self.output_dir}/policies/policy-{self.step}.pth",
        )
        self.agent.policy_network.to(self.device)
        torch.save(
            self.agent.critic_network.to("cpu").state_dict(),
            f"{self.output_dir}/critics/critic-{self.step}-{self.experiment_name}.pth",
        )
        self.agent.critic_network.to(self.device)

    def _guess_save_path_model(self) -> dict[str]:
        return {
            "policy": f"{self.output_dir}/policies/policy-{self.step}.pth",
            "critic": f"{self.output_dir}/critics/critic-{self.step}-{self.experiment_name}.pth",
        }

    def sample_omega(self, step):
        """Sample uncertainty parameters

        Parameters
        ----------
        step : int
            current training step

        """
        if step <= self.start_steps:
            omega = self.rand_state.uniform(
                low=self.change_param_min,
                high=self.change_param_max,
                size=len(self.change_param_min),
            )
            dis_restart_flag = "None"
            prob_restart_flag = "None"
        else:
            omega, dis_restart_flag, prob_restart_flag = self.agent.get_omega()
            if dis_restart_flag:
                dis_restart_flag = "True"
            else:
                dis_restart_flag = "False"
            if prob_restart_flag:
                prob_restart_flag = "True"
            else:
                prob_restart_flag = "False"

        assert len(omega) == len(self.change_param_min) == len(self.change_param_max)

        omega = np.clip(
            omega,
            self.change_param_min,
            self.change_param_max,
        )
        assert isinstance(omega, np.ndarray)

        return omega, dis_restart_flag, prob_restart_flag

    def interact(self, env, omega):
        """Interaction environment with omega

        parameters
        ----------
        env : gym.Env
            gym environment
        omega : np.Array
            Uncertainty parameters defining the environment

        """
        omega_dict = {k: v for k, v in zip(self.uncertainty_set.keys(), omega)}
        state, _ = env.reset(seed=self.seed, options=omega_dict)
        total_reward = 0
        done, truncated = False, False
        while not done and not truncated:
            if self.step <= self.start_steps:
                action = self.rand_state.uniform(
                    low=env.action_space.low,
                    high=env.action_space.high,
                    size=env.action_space.low.shape,
                ).astype(env.action_space.low.dtype)
            else:
                action = self.agent.get_action(state)

            next_state, reward, done, truncated, _ = env.step(action)

            self.agent.add_memory(state, action, next_state, reward, done, omega)

            if self.step >= self.start_steps:
                self.agent.train(self.step)

            state = next_state
            total_reward += reward
            if done or truncated:
                wandb.log({"episode reward": total_reward}, step=self.step)

            if self.step % self.evaluate_qvalue_interval == 0:
                pass
            if self.step % self.logger_interval == 0:
                pass
            if self.step % self.evaluate_interval == 0:
                self.save_model()
            if self.step >= self.max_steps:
                return True, None

            self.step += 1
            self.episode_len += 1
            if done or truncated:
                return False, total_reward

        return False, total_reward

    def main(self):
        """Training"""
        while True:
            self.agent.set_current_episode_len(self.episode_len)
            self.episode_len = 0
            omega, _, _ = self.sample_omega(self.step)

            flag, total_reward = self.interact(self.env, omega)
            print(f"Step: {self.step} Omega: {omega} Total reward: {total_reward}")
            if flag:
                break
        self.save_model()
        path_model = self._guess_save_path_model()
        # save file in the artifact for the experiment
        m2td3_artifact = wandb.Artifact(
            "model",
            type="model",
            metadata={"step": self.step},
            description="model",
        )
        m2td3_artifact.add_file(path_model["policy"], "policy.pth")
        m2td3_artifact.add_file(path_model["critic"], "critic.pth")
        wandb.log_artifact(m2td3_artifact)
