from typing import Annotated, Any, Dict, List, Self
import gymnasium as gym
import numpy as np
import rrls  # type: ignore  # noqa: F401
from rrls._interface import ModifiedParamsEnv
from rrls.envs import (  # type: ignore
    AntParamsBound,
    HalfCheetahParamsBound,
    HopperParamsBound,
    HumanoidStandupParamsBound,
    InvertedPendulumParamsBound,
    Walker2dParamsBound,
)
import torch

from m2td3.agent_wrapper import M2TD3AgentWrapper
from evaluation import Agent
from td3.models import Actor, QNetwork
from td3.td3 import TD3Agent
from mock_agent import MockAgent
from scheduler import (
    Scheduler,
    LinearScheduler,
    ExponentialScheduler,
    LogarithmicScheduler,
    RandomScheduler,
    CosineScheduler,
)


ENV_NAME = {
    "Ant",
    "HalfCheetah",
    "Hopper",
    "HumanoidStandup",
    "InvertedPendulum",
    "Walker",
}

BOUNDS = {
    "Ant": AntParamsBound,
    "HalfCheetah": HalfCheetahParamsBound,
    "Hopper": HopperParamsBound,
    "HumanoidStandup": HumanoidStandupParamsBound,
    "InvertedPendulum": InvertedPendulumParamsBound,
    "Walker": Walker2dParamsBound,
}


def env_factory(env_name: str) -> gym.Env:
    ENV_IDENTIFIERS = {
        "Ant": "rrls/robust-ant-v0",
        "HalfCheetah": "rrls/robust-halfcheetah-v0",
        "Hopper": "rrls/robust-hopper-v0",
        "HumanoidStandup": "rrls/robust-humanoidstandup-v0",
        "InvertedPendulum": "rrls/robust-invertedpendulum-v0",
        "Walker": "rrls/robust-walker-v0",
    }
    env_id = ENV_IDENTIFIERS.get(env_name)
    if env_id is not None:
        return gym.make(env_id)
    else:
        raise ValueError(f"Environment '{env_name}' is not supported.")


class DuplicateObservation(gym.ObservationWrapper):
    # THIS IS AN HELPER FOR THE RARL ALGORITHM, IN OUR FRAMEWORK OBSERVATION IS A TUPLE,
    # FOR THE AGENT AND THE ADVERSAIRY.
    def __init__(
        self,
        env: gym.Env,
        omniscient_adversary: bool = False,
    ):
        super().__init__(env)
        adversary_observation_size: int = env.observation_space.shape[0]
        if omniscient_adversary:
            adversary_observation_size += env.action_space[0].shape[0]

        adversary_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(adversary_observation_size,),
        )

        self.observation_space = gym.spaces.Tuple(
            spaces=(env.observation_space, adversary_observation_space)
        )

    def observation(self, observation):
        # HACK: In case of omniscient adversary, we need to add the action to the observation
        # to make it compatible with the observation space but we don't do it because
        # we get the action  from the agent and we concatenate it with the observation
        return observation, observation


def bound_factory(env_name, nb_dim: int) -> Dict[str, Annotated[List[float], 2]]:
    bound = BOUNDS[env_name]
    if nb_dim == 3:
        return bound.THREE_DIM.value
    if nb_dim == 2:
        return bound.TWO_DIM.value
    return bound.ONE_DIM.value


def scheduler_factory(
    scheduler_type: str, params_bound: Dict[str, Annotated[List[float], 2]], **kwargs
) -> Scheduler:
    match scheduler_type:
        case "linear":
            return LinearScheduler(params_bound=params_bound, **kwargs)
        case "exponential":
            return ExponentialScheduler(params_bound=params_bound, **kwargs)
        case "logarithmic":
            return LogarithmicScheduler(params_bound=params_bound, **kwargs)
        case "random":
            return RandomScheduler(params_bound=params_bound, **kwargs)
        case "cosine":
            return CosineScheduler(params_bound=params_bound, **kwargs)
        case _:
            raise ValueError(f"Scheduler type '{scheduler_type}' is not supported.")


class AgentInferenceBuilder:
    def __init__(
        self,
        env: str | gym.Env,
        nb_dim: int,
        device: str,
        env_wrapper: gym.Wrapper = None,
    ) -> None:
        # self.env_name: str = env_name
        self.nb_dim: int = nb_dim
        if isinstance(env, str):
            self.env_name = env
            self.env = env_factory(env_name=env)
        else:
            self.env_name = env.unwrapped.spec.id
            self.env = env
        # self.env = env_factory(env_name=env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def add_actor_path(self, path: str) -> Self:
        self.path = path
        return self

    def add_device(self, device: str) -> Self:
        self.device = torch.device(device)
        return self

    def add_agent_type(self, agent_type: str) -> Self:
        self.agent_type = agent_type
        return self

    def build(self) -> Agent:
        obs_dim: int = np.prod(self.observation_space.shape)
        action_dim: int = np.prod(self.action_space.shape)
        match self.agent_type:
            case "m2td3":
                # print("building m2td3 agent")
                return M2TD3AgentWrapper(
                    policy_path=self.path,
                    state_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_num=1,
                    hidden_layer=256,  # TODO: Fix those hardcoded values
                    max_action=self.action_space.high[0],
                    device=self.device,
                )
            case "td3":
                # print("building td3 agent")
                actor = Actor(
                    observation_dim=obs_dim,
                    action_space=self.action_space,
                )
                actor.load_state_dict(
                    state_dict=torch.load(self.path, map_location=self.device)["actor"]
                )
                actor.to(device=self.device)
                qf1 = QNetwork(
                    observation_dim=obs_dim,
                    action_dim=action_dim,
                ).to(device=self.device)

                qf2 = QNetwork(
                    observation_dim=obs_dim,
                    action_dim=action_dim,
                ).to(device=self.device)
                return TD3Agent(
                    actor=actor,
                    qf1=qf1,
                    qf2=qf2,
                    device=self.device,
                )
            case "mock":
                return MockAgent(action_space=self.action_space)
            case _:
                raise ValueError(f"Agent type '{self.agent_type}' is not supported.")


class ParametersObservable(gym.Wrapper):
    def __init__(self, env: ModifiedParamsEnv, params_bound: dict[str, tuple[float]]):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(env.observation_space.shape[0] + len(params_bound),),
        )
        self.params_bound = params_bound
        env.set_params()

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs: np.ndarray
        info: dict[str, float]
        obs, info = self.env.reset(seed=seed, options=options)
        params: dict[str, float] = self.env.get_params()
        filtred_params = {k: v for k, v in params.items() if k in self.params_bound}
        params_obs = np.fromiter(filtred_params.values(), dtype=float)
        obs = np.concatenate((obs, params_obs))
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, done, truncated, info = self.env.step(action)
        params: dict[str, float] = self.env.get_params()
        filtred_params = {k: v for k, v in params.items() if k in self.params_bound}
        params_obs = np.fromiter(filtred_params.values(), dtype=float)
        obs = np.concatenate((obs, params_obs))
        return obs, reward, done, truncated, info

    def set_params(self, **kwargs):
        self.env.set_params(**kwargs)

    def get_params(self):
        return self.env.get_params()
