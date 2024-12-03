from typing import Annotated, Any, SupportsFloat
import gymnasium as gym
import numpy as np
from rrls._interface import ModifiedParamsEnv
from typing import Protocol


class Agent(Protocol):
    def select_action(
        self, obs: np.ndarray, use_random: bool = False
    ) -> np.ndarray: ...


class TCMDP(gym.Wrapper):
    """
    A wrapper class for a modified version of the TC-MDP environment.

    Args:
        env (ModifiedParamsEnv): The modified version of the TC-MDP environment.
        params_bound (dict[str, Annotated[tuple[float], 2]]): The bounds for the parameters.
        radius (float, optional): The radius for the adversarial action. Defaults to 0.001.
    """

    def __init__(
        self,
        env: ModifiedParamsEnv,
        params_bound: dict[str, Annotated[tuple[float], 2]],
        radius: float = 0.001,
        omniscient_adversary: bool = False,
    ):
        super().__init__(env=env)

        # s + \psi
        adversary_observation_size: int = np.sum(env.observation_space.shape) + len(
            params_bound
        )
        if omniscient_adversary:
            # Change the observation space to include to the adversary observation space
            # the action of the agent.
            # now s + \psi + a
            adversary_observation_size += env.action_space.shape[0]

        self.observation_space = gym.spaces.Tuple(
            spaces=(
                env.observation_space,
                gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(adversary_observation_size,),
                ),
            )
        )

        self.action_space = gym.spaces.Tuple(
            spaces=(
                env.action_space,
                gym.spaces.Box(
                    low=-1, high=1, shape=(len(params_bound),)
                ),  # adversarial action
            )
        )
        self.params_bound: dict[str, tuple[float]] = params_bound
        self.defaut_params: dict[str, float] = env.get_params()
        self.radius = radius
        self.env = env

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int | None, optional): The random seed for the environment. Defaults to None.
            options (dict[str, Any] | None, optional): Additional options for the environment. Defaults to None.

        Returns:
            tuple[Any, dict[str, Any]]: The initial observation and additional information.
        """
        new_params: dict[str, float] = self._draw_params()
        if options is None:
            options = {}
        options = {**options, **new_params}
        self.current_params = new_params
        obs, info = self.env.reset(seed=seed, options=options)
        info["current_params"] = self.current_params
        psi = {k: self.current_params[k] for k in self.params_bound.keys()}
        info["psi"] = psi
        adv_obs = np.fromiter(psi.values(), dtype=float)
        adv_obs = np.concatenate((adv_obs, obs))
        obs = (obs, adv_obs)
        return obs, info

    def _draw_params(self) -> dict[str, float]:
        """
        Draw random parameters within the specified bounds.

        Returns:
            dict[str, float]: The randomly drawn parameters.
        """
        return {k: self.np_random.uniform(*v) for k, v in self.params_bound.items()}

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action (Any): The action to take in the environment.

        Returns:
            tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]: The next observation, reward, done flag, truncated flag, and additional information.
        """
        action, adversarial_action = action
        self.current_params: dict[str, float] = self._computes_new_params(
            action=adversarial_action
        )
        # The parameters of the environment are updated before the agent action
        self.env.set_params(**self.current_params)

        obs, reward, done, truncated, info = self.env.step(action)
        info["current_params"] = self.current_params
        psi = {k: self.current_params[k] for k in self.params_bound.keys()}
        info["psi"] = psi
        adv_obs = np.fromiter(psi.values(), dtype=float)
        adv_obs = np.concatenate((adv_obs, obs))
        obs = (obs, adv_obs)
        return obs, reward, done, truncated, info

    def _computes_new_params(self, action: np.ndarray) -> dict[str, float]:
        """
        Compute the new parameters based on the adversarial action.

        Args:
            action (np.ndarray): The adversarial action.

        Returns:
            dict[str, float]: The new parameters.
        """
        action_dict: dict[str, float] = {
            k: v for k, v in zip(self.params_bound.keys(), action)
        }
        # We clip the new parameters to the bounds
        new_params = {
            k: np.clip(v * self.radius + self.current_params[k], *self.params_bound[k])
            for k, v in action_dict.items()
        }
        return new_params


class TCMDPFixedAgent(gym.Wrapper):
    """
    A wrapper class for a modified version of the TC-MDP environment with a fixed adversary.

    Args:
        env (ModifiedParamsEnv): The modified version of the TC-MDP environment.
        params_bound (dict[str, Annotated[tuple[float], 2]]): The bounds for the parameters.
        radius (float, optional): The radius for the adversarial action. Defaults to 0.1.
    """

    def __init__(
        self,
        env: ModifiedParamsEnv,
        agent: Agent,
        params_bound: dict[str, Annotated[tuple[float], 2]],
        radius: float = 0.1,
        is_omniscient: bool = False,
    ):
        super().__init__(env=env)
        self.agent: Agent = agent

        adversary_observation_size: int = np.sum(env.observation_space.shape) + len(
            params_bound
        )
        if is_omniscient:
            # Change the observation space to include to the adversary observation space
            # the action of the agent.
            adversary_observation_size += env.action_space.shape[0]
        self.is_omniscient = is_omniscient

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(adversary_observation_size,),
        )

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(len(params_bound),)
        )  # adversarial action
        self.params_bound: dict[str, tuple[float]] = params_bound
        self.defaut_params = env.get_params()
        self.radius = radius
        self.env = env
        self.obs_agent = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int | None, optional): The random seed for the environment. Defaults to None.
            options (dict[str, Any] | None, optional): Additional options for the environment. Defaults to None.

        Returns:
            tuple[Any, dict[str, Any]]: The initial observation and additional information.
        """
        new_params: dict[str, float] = self._draw_params()
        if options is None:
            options = {}
        options = {**options, **new_params}
        self.current_params = new_params
        obs: np.ndarray
        info: dict[str, float]
        obs, info = self.env.reset(seed=seed, options=options)

        info["current_params"] = self.current_params
        adv_obs = np.fromiter(info["current_params"].values(), dtype=float)
        adv_obs = np.concatenate((adv_obs, obs))
        obs = (obs, adv_obs)
        self.obs_agent = obs[0]

        if self.is_omniscient:
            action_agent = self.agent.select_action(self.obs_agent, use_random=False)
            adv_obs = np.concatenate((adv_obs, action_agent))
        return adv_obs, info

    def _draw_params(self) -> dict[str, float]:
        """
        Draw random parameters within the specified bounds.

        Returns:
            dict[str, float]: The randomly drawn parameters.
        """
        return {k: self.np_random.uniform(*v) for k, v in self.params_bound.items()}

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action (Any): The action to take in the environment.

        Returns:
            tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]: The next observation, reward, done flag, truncated flag, and additional information.
        """
        adversarial_action = action
        # TODO: Optimize this part with caching the previous action
        agent_action = self.agent.select_action(self.obs_agent, use_random=False)

        self.current_params: dict[str, float] = self._computes_new_params(
            action=adversarial_action
        )
        self.env.set_params(**self.current_params)
        obs: np.ndarray
        obs, reward, done, truncated, info = self.env.step(agent_action)

        info["current_params"] = self.current_params
        adv_obs = np.fromiter(info["current_params"].values(), dtype=float)
        adv_obs = np.concatenate((adv_obs, obs))

        self.obs_agent = obs

        if self.is_omniscient:
            action_agent = self.agent.select_action(self.obs_agent, use_random=False)
            adv_obs = np.concatenate((adv_obs, action_agent))

        return adv_obs, -reward, done, truncated, info

    def _computes_new_params(self, action: np.ndarray) -> dict[str, float]:
        """
        Compute the new parameters based on the adversarial action.

        Args:
            action (np.ndarray): The adversarial action.

        Returns:
            dict[str, float]: The new parameters.
        """
        action_dict: dict[str, float] = {
            k: v for k, v in zip(self.params_bound.keys(), action)
        }
        # We clip the new parameters to the bounds
        new_params: dict[str, float] = {
            k: np.clip(v * self.radius + self.current_params[k], *self.params_bound[k])
            for k, v in action_dict.items()
        }
        return new_params


class OracleTCMDP(TCMDP):
    """
    A wrapper class for a modified version of the TC-MDP environment with an oracle agent that knows the true parameters.

    Args:
        env (ModifiedParamsEnv): The modified version of the TC-MDP environment.
        params_bound (dict[str, Annotated[tuple[float], 2]]): The bounds for the parameters.
        radius (float, optional): The radius for the adversarial action. Defaults to 0.1.
    """

    def __init__(
        self,
        env: ModifiedParamsEnv,
        params_bound: dict[str, Annotated[tuple[float], 2]],
        radius: float = 0.1,
        omniscient_adversary: bool = False,
    ):
        super().__init__(
            env=env,
            params_bound=params_bound,
            radius=radius,
            omniscient_adversary=omniscient_adversary,
        )
        observation_size: int = (
            np.sum(env.observation_space.shape) + len(params_bound)
        )  # An observation is the concatenation of the current observation and the psi values
        if omniscient_adversary:
            # Change the observation space to include to the adversary observation space
            # the action of the agent.
            # now s + \psi + a
            adversary_observation_size = observation_size + env.action_space.shape[0]

        self.observation_space = gym.spaces.Tuple(
            spaces=(
                gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(observation_size,),
                ),
                gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(adversary_observation_size,),
                ),
            )
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int | None, optional): The random seed for the environment. Defaults to None.
            options (dict[str, Any] | None, optional): Additional options for the environment. Defaults to None.

        Returns:
            tuple[Any, dict[str, Any]]: The initial observation and additional information.
        """
        new_params: dict[str, float] = self._draw_params()
        if options is None:
            options = {}
        options = {**options, **new_params}
        self.current_params = new_params
        obs, info = self.env.reset(seed=seed, options=options)
        info["current_params"] = self.current_params
        psi = {
            k: (self.current_params[k] - self.params_bound[k][0])
            / (self.params_bound[k][1] - self.params_bound[k][0])
            for k in self.params_bound.keys()
        }
        info["psi"] = psi
        psi_obs = np.fromiter(psi.values(), dtype=float)
        # obs = np.concatenate((psi_obs, obs))
        obs = np.concatenate((obs, psi_obs))
        obs = (obs, obs)
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action (Any): The action to take in the environment.

        Returns:
            tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]: The next observation, reward, done flag, truncated flag, and additional information.
        """
        action, adversarial_action = action
        self.current_params: dict[str, float] = self._computes_new_params(
            action=adversarial_action
        )
        # The parameters of the environment are updated before the agent action
        self.env.set_params(**self.current_params)

        obs, reward, done, truncated, info = self.env.step(action)
        info["current_params"] = self.current_params
        psi = {
            k: (self.current_params[k] - self.params_bound[k][0])
            / (self.params_bound[k][1] - self.params_bound[k][0])
            for k in self.params_bound.keys()
        }
        info["psi"] = psi
        psi_obs = np.fromiter(psi.values(), dtype=float)
        # obs = np.concatenate((psi_obs, obs))
        obs = np.concatenate((obs, psi_obs))
        obs = (obs, obs)
        return obs, reward, done, truncated, info


class EvalOracleTCMDP(gym.Wrapper):
    """
    A wrapper class for a modified version of the TC-MDP environment where the agent has access to the true parameters.

    Args:
        env (ModifiedParamsEnv): The modified version of the TC-MDP environment.
        params_bound (dict[str, Annotated[tuple[float], 2]]): The bounds for the parameters.
    """

    def __init__(
        self,
        env: ModifiedParamsEnv,
        params_bound: dict[str, Annotated[tuple[float], 2]],
    ):
        super().__init__(env=env)

        # s + \psi
        observation_size: int = np.sum(env.observation_space.shape) + len(params_bound)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
        )

        self.action_space = env.action_space

        self.params_bound: dict[str, tuple[float]] = params_bound
        self.defaut_params: dict[str, float] = env.get_params()

    def set_params(self, **new_params):
        self.env.set_params(**new_params)
        self.current_params = new_params

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int | None, optional): The random seed for the environment. Defaults to None.
            options (dict[str, Any] | None, optional): Additional options for the environment. Defaults to None.

        Returns:
            tuple[Any, dict[str, Any]]: The initial observation and additional information.
        """
        new_params: dict[str, float] = self._draw_params()
        if options is None:
            options = {}
        options = {**options, **new_params}
        self.current_params = new_params
        obs, info = self.env.reset(seed=seed, options=options)
        info["current_params"] = self.current_params
        psi = {
            k: (self.current_params[k] - self.params_bound[k][0])
            / (self.params_bound[k][1] - self.params_bound[k][0])
            for k in self.params_bound.keys()
        }
        info["psi"] = psi
        psi_obs = np.fromiter(psi.values(), dtype=float)
        obs = np.concatenate((obs, psi_obs))
        # obs = np.concatenate((psi_obs, obs))
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action (Any): The action to take in the environment.

        Returns:
            tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]: The next observation, reward, done flag, truncated flag, and additional information.
        """
        obs, reward, done, truncated, info = self.env.step(action)
        info["current_params"] = self.current_params
        psi = {
            k: (self.current_params[k] - self.params_bound[k][0])
            / (self.params_bound[k][1] - self.params_bound[k][0])
            for k in self.params_bound.keys()
        }
        info["psi"] = psi
        psi_obs = np.fromiter(psi.values(), dtype=float)
        obs = np.concatenate((obs, psi_obs))
        # obs = np.concatenate((psi_obs, obs))
        return obs, reward, done, truncated, info

    def _draw_params(self) -> dict[str, float]:
        """
        Draw random parameters within the specified bounds.

        Returns:
            dict[str, float]: The randomly drawn parameters.
        """
        return {k: self.np_random.uniform(*v) for k, v in self.params_bound.items()}

    def get_params(self) -> dict[str, float]:
        """
        Get the current parameters of the environment.

        Returns:
            dict[str, float]: The current parameters of the environment.
        """
        return self.env.get_params()


class StackedTCMDP(TCMDP):
    """
    A wrapper class for a modified version of the TC-MDP environment with a stacked observation.

    Args:
        env (ModifiedParamsEnv): The modified version of the TC-MDP environment.
        params_bound (dict[str, Annotated[tuple[float], 2]]): The bounds for the parameters.
        radius (float, optional): The radius for the adversarial action. Defaults to 0.1.
    """

    def __init__(
        self,
        env: ModifiedParamsEnv,
        params_bound: dict[str, Annotated[tuple[float], 2]],
        radius: float = 0.1,
        omniscient_adversary: bool = False,
    ):
        super().__init__(
            env=env,
            params_bound=params_bound,
            radius=radius,
            omniscient_adversary=omniscient_adversary,
        )
        observation_size: int = (
            np.sum(env.observation_space.shape) * 2
            + env.action_space.shape[
                0
            ]  # An observation is the concatenation of the previous observation, the current observation, and the previous action
        )
        adversary_observation_size: int = (
            np.sum(env.observation_space.shape) + len(params_bound)
        )  # An adversary observation is the concatenation of the current observation and the psi values
        if omniscient_adversary:
            # Change the observation space to include to the adversary observation space
            # the action of the agent.
            # now s + \psi + a
            adversary_observation_size += env.action_space.shape[0]
        self.observation_space = gym.spaces.Tuple(
            spaces=(
                gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(observation_size,),
                ),
                gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(adversary_observation_size,),
                ),
            )
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int | None, optional): The random seed for the environment. Defaults to None.
            options (dict[str, Any] | None, optional): Additional options for the environment. Defaults to None.

        Returns:
            tuple[Any, dict[str, Any]]: The initial observation and additional information.
        """
        new_params: dict[str, float] = self._draw_params()
        if options is None:
            options = {}
        options = {**options, **new_params}
        self.current_params = new_params
        obs, info = self.env.reset(seed=seed, options=options)
        info["current_params"] = self.current_params
        psi = {
            k: (self.current_params[k] - self.params_bound[k][0])
            / (self.params_bound[k][1] - self.params_bound[k][0])
            for k in self.params_bound.keys()
        }
        info["psi"] = psi
        adv_obs = np.fromiter(psi.values(), dtype=float)
        adv_obs = np.concatenate((adv_obs, obs))
        self.prev_obs = obs.copy()
        action = np.zeros(self.env.action_space.shape[0])
        obs = (np.concatenate((self.prev_obs, obs, action)), adv_obs)
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action (Any): The action to take in the environment.

        Returns:
            tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]: The next observation, reward, done flag, truncated flag, and additional information.
        """
        action, adversarial_action = action
        self.current_params: dict[str, float] = self._computes_new_params(
            action=adversarial_action
        )
        # The parameters of the environment are updated before the agent action
        self.env.set_params(**self.current_params)

        obs, reward, done, truncated, info = self.env.step(action)
        info["current_params"] = self.current_params
        psi = {
            k: (self.current_params[k] - self.params_bound[k][0])
            / (self.params_bound[k][1] - self.params_bound[k][0])
            for k in self.params_bound.keys()
        }
        info["psi"] = psi
        adv_obs = np.fromiter(psi.values(), dtype=float)
        adv_obs = np.concatenate((adv_obs, obs))
        staked_obs = np.concatenate((self.prev_obs.copy(), obs, action))
        self.prev_obs = obs.copy()
        obs = (staked_obs, adv_obs)
        return obs, reward, done, truncated, info


class EvalStackedTCMDP(TCMDP):
    """
    A wrapper class for a modified version of the TC-MDP environment with a stacked observation.

    Args:
        env (ModifiedParamsEnv): The modified version of the TC-MDP environment.
        params_bound (dict[str, Annotated[tuple[float], 2]]): The bounds for the parameters.
        radius (float, optional): The radius for the adversarial action. Defaults to 0.1.
    """

    def __init__(
        self,
        env: ModifiedParamsEnv,
        params_bound: dict[str, Annotated[tuple[float], 2]],
        radius: float = 0.1,
        omniscient_adversary: bool = False,
    ):
        super().__init__(
            env=env,
            params_bound=params_bound,
            radius=radius,
            omniscient_adversary=omniscient_adversary,
        )
        observation_size: int = (
            np.sum(env.observation_space.shape) * 2
            + env.action_space.shape[
                0
            ]  # An observation is the concatenation of the previous observation, the current observation, and the previous action
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
        )

        self.action_space = env.action_space

    def set_params(self, **new_params):
        self.env.set_params(**new_params)
        self.current_params = new_params

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int | None, optional): The random seed for the environment. Defaults to None.
            options (dict[str, Any] | None, optional): Additional options for the environment. Defaults to None.

        Returns:
            tuple[Any, dict[str, Any]]: The initial observation and additional information.
        """
        new_params: dict[str, float] = self._draw_params()
        if options is None:
            options = {}
        options = {**options, **new_params}
        self.current_params = new_params
        obs, info = self.env.reset(seed=seed, options=options)
        info["current_params"] = self.current_params
        psi = {
            k: (self.current_params[k] - self.params_bound[k][0])
            / (self.params_bound[k][1] - self.params_bound[k][0])
            for k in self.params_bound.keys()
        }
        info["psi"] = psi
        self.prev_obs = obs.copy()
        action = np.zeros(self.env.action_space.shape[0])
        obs = np.concatenate((self.prev_obs, obs, action))
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action (Any): The action to take in the environment.

        Returns:
            tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]: The next observation, reward, done flag, truncated flag, and additional information.
        """
        # The parameters of the environment are updated before the agent action
        self.env.set_params(**self.current_params)

        obs, reward, done, truncated, info = self.env.step(action)
        info["current_params"] = self.current_params
        psi = {
            k: (self.current_params[k] - self.params_bound[k][0])
            / (self.params_bound[k][1] - self.params_bound[k][0])
            for k in self.params_bound.keys()
        }
        info["psi"] = psi
        staked_obs = np.concatenate((self.prev_obs.copy(), obs, action))
        self.prev_obs = obs.copy()
        return staked_obs, reward, done, truncated, info

    def get_params(self) -> dict[str, float]:
        """
        Get the current parameters of the environment.

        Returns:
            dict[str, float]: The current parameters of the environment.
        """
        return self.env.get_params()


class EvalStaticOracle(gym.Wrapper):
    """
    A wrapper class for a modified version of the environment where the agent has access to the true parameters.

    Args:
        env (ModifiedParamsEnv): An robust env.
        params_bound (dict[str, Annotated[tuple[float], 2]]): The bounds for the parameters.
    """

    def __init__(
        self,
        env: ModifiedParamsEnv,
        params_bound: dict[str, Annotated[tuple[float], 2]],
    ):
        super().__init__(env=env)

        # s + \psi
        observation_size: int = np.sum(env.observation_space.shape) + len(params_bound)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
        )

        self.action_space = env.action_space

        self.params_bound: dict[str, tuple[float]] = params_bound
        self.defaut_params: dict[str, float] = env.get_params()
        self.current_params = self.defaut_params

    def set_params(self, **new_params):
        self.env.set_params(**new_params)
        self.current_params = new_params

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int | None, optional): The random seed for the environment. Defaults to None.
            options (dict[str, Any] | None, optional): Additional options for the environment. Defaults to None.

        Returns:
            tuple[Any, dict[str, Any]]: The initial observation and additional information.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        info["current_params"] = self.current_params

        # normalized psi
        psi = {
            k: (self.current_params[k] - self.params_bound[k][0])
            / (self.params_bound[k][1] - self.params_bound[k][0])
            for k in self.params_bound.keys()
        }
        info["psi"] = psi

        psi_obs = np.fromiter(psi.values(), dtype=float)
        obs = np.concatenate((obs, psi_obs))
        # obs = np.concatenate((psi_obs, obs))
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action (Any): The action to take in the environment.

        Returns:
            tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]: The next observation, reward, done flag, truncated flag, and additional information.
        """
        obs, reward, done, truncated, info = self.env.step(action)
        info["current_params"] = self.current_params
        psi = {
            k: (self.current_params[k] - self.params_bound[k][0])
            / (self.params_bound[k][1] - self.params_bound[k][0])
            for k in self.params_bound.keys()
        }
        info["psi"] = psi
        psi_obs = np.fromiter(psi.values(), dtype=float)
        obs = np.concatenate((obs, psi_obs))
        # obs = np.concatenate((psi_obs, obs))
        return obs, reward, done, truncated, info

    def get_params(self) -> dict[str, float]:
        """
        Get the current parameters of the environment.

        Returns:
            dict[str, float]: The current parameters of the environment.
        """
        return self.env.get_params()


class EvalStaticStacked(gym.Wrapper):
    """
    A wrapper class for a modified version of the TC-MDP environment with a stacked observation.

    Args:
        env (ModifiedParamsEnv): The modified version of the TC-MDP environment.
        params_bound (dict[str, Annotated[tuple[float], 2]]): The bounds for the parameters.
        radius (float, optional): The radius for the adversarial action. Defaults to 0.1.
    """

    def __init__(
        self,
        env: ModifiedParamsEnv,
        params_bound: dict[str, Annotated[tuple[float], 2]],
    ):
        super().__init__(
            env=env,
        )
        observation_size: int = (
            np.sum(env.observation_space.shape) * 2
            + env.action_space.shape[
                0
            ]  # An observation is the concatenation of the previous observation, the current observation, and the previous action
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
        )

        self.action_space = env.action_space
        self.params_bound = params_bound
        # NOTE: It will never change during the episode because we are static for this evaluation
        self.defaut_params: dict[str, float] = env.get_params()
        self.current_params = self.defaut_params

    def set_params(self, **new_params):
        self.env.set_params(**new_params)
        self.current_params = new_params

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int | None, optional): The random seed for the environment. Defaults to None.
            options (dict[str, Any] | None, optional): Additional options for the environment. Defaults to None.

        Returns:
            tuple[Any, dict[str, Any]]: The initial observation and additional information.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        info["current_params"] = self.current_params
        self.prev_obs = obs.copy()
        action = np.zeros(self.env.action_space.shape[0])
        obs = np.concatenate((self.prev_obs, obs, action))
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action (Any): The action to take in the environment.

        Returns:
            tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]: The next observation, reward, done flag, truncated flag, and additional information.
        """
        # The parameters of the environment are updated before the agent action
        self.env.set_params(**self.current_params)

        obs, reward, done, truncated, info = self.env.step(action)
        info["current_params"] = self.current_params
        psi = {
            k: (self.current_params[k] - self.params_bound[k][0])
            / (self.params_bound[k][1] - self.params_bound[k][0])
            for k in self.params_bound.keys()
        }
        info["psi"] = psi
        staked_obs = np.concatenate((self.prev_obs.copy(), obs, action))
        self.prev_obs = obs.copy()
        return staked_obs, reward, done, truncated, info

    def get_params(self) -> dict[str, float]:
        """
        Get the current parameters of the environment.

        Returns:
            dict[str, float]: The current parameters of the environment.
        """
        return self.env.get_params()
