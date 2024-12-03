from typing import Annotated, Dict, List
import gymnasium as gym
import rrls
from rrls.envs import (
    AntParamsBound,
    HalfCheetahParamsBound,
    HopperParamsBound,
    HumanoidStandupParamsBound,
    InvertedPendulumParamsBound,
    Walker2dParamsBound,
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
    ENV_FACTORY = {
        "Ant": gym.make("rrls/robust-ant-v0"),
        "HalfCheetah": gym.make("rrls/robust-halfcheetah-v0"),
        "Hopper": gym.make("rrls/robust-hopper-v0"),
        "HumanoidStandup": gym.make("rrls/robust-humanoidstandup-v0"),
        "InvertedPendulum": gym.make("rrls/robust-invertedpendulum-v0"),
        "Walker": gym.make("rrls/robust-walker-v0"),
    }
    return ENV_FACTORY[env_name]


def bound_factory(env_name, nb_dim: int) -> Dict[str, Annotated[List[float], 2]]:
    bound = BOUNDS[env_name]
    if nb_dim == 3:
        return bound.THREE_DIM.value
    if nb_dim == 2:
        return bound.TWO_DIM.value
    return bound.ONE_DIM.value
