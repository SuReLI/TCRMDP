from typing import Union
import numpy as np
from abc import ABC

ParamBound = dict[str, list[float]]


class Scheduler(ABC):
    """
    Abstract base class for schedulers.
    """

    def __init__(self, params_bound: ParamBound, seed: int | None = None) -> None:
        self.params_bound = params_bound
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> None:
        """
        Abstract method to reset the scheduler.

        Args:
            seed (int or None, optional): The seed value for random number generation. Defaults to None.
        """
        raise NotImplementedError()

    def sample(self, t: int) -> dict[str, float]:
        """
        Abstract method to sample parameters for a given time step.

        Args:
            t (int): The time step.

        Returns:
            dict[str, float]: A dictionary of sampled parameters.
        """
        raise NotImplementedError()


class PrecomputedScheduler(Scheduler):
    """
    Base class for precomputed schedulers.
    """

    def __init__(
        self,
        params_bound: ParamBound,
        max_step: int = 1000,
        seed: int | None = None,
    ) -> None:
        self.max_step = max_step
        super().__init__(params_bound=params_bound, seed=seed)
        seed = self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        """
        Reset the scheduler.

        Args:
            seed (int or None, optional): The seed value for random number generation. Defaults to None.
        """

        if seed is None:
            seed = np.random.randint(0, 1000)
        self.rng = np.random.default_rng(seed=seed)
        self._init_params(params_bound=self.params_bound, max_step=self.max_step)

    def _init_params(self, params_bound: ParamBound, max_step: int) -> None:
        """
        Initialize the parameters.

        Args:
            params_bound (ParamBound): The parameter bounds.
            max_step (int): The maximum number of steps.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """

        raise NotImplementedError()

    def sample(self, t: int) -> dict[str, float]:
        """
        Sample the parameters at a given time step.

        Args:
            t (int): The time step.

        Returns:
            dict[str, float]: The sampled parameters.
        """
        return {k: v[t] for k, v in self.params.items()}


class LinearScheduler(PrecomputedScheduler):
    """
    A scheduler that samples parameters linearly.
    Goes from the start point to the direction point in a linear fashion.
    """

    def __init__(
        self,
        params_bound: ParamBound,
        max_step: int = 1000,
        seed: Union[int, None] = None,
    ) -> None:
        """
        Initializes a LinearScheduler object.

        Args:
            params_bound (ParamBound): A dictionary specifying the parameter bounds.
            max_step (int, optional): The maximum number of steps. Defaults to 1000.
            seed (int or None, optional): The seed value for random number generation. Defaults to None.
        """
        super().__init__(params_bound=params_bound, max_step=max_step, seed=seed)

    def _init_params(self, params_bound: ParamBound, max_step: int):
        """
        Initializes the parameters for the linear scheduler.

        Args:
            params_bound (ParamBound): A dictionary specifying the parameter bounds.
            max_step (int): The maximum number of steps before reaching the last value.
        """
        self.params = {}
        for k, v in params_bound.items():
            assert len(v) == 2 and v[0] < v[1]
            start_point = self.rng.uniform(v[0], v[1])
            direction = self.rng.choice([v[0], v[1]])
            self.params[k] = np.linspace(start_point, direction, max_step)


class ExponentialScheduler(PrecomputedScheduler):
    """
    A scheduler that samples parameters exponentially.
    Goes from the start point to the direction point in an exponential fashion.
    """

    def __init__(
        self,
        params_bound: ParamBound,
        max_step: int = 1000,
        seed: int | None = None,
    ) -> None:
        super().__init__(params_bound=params_bound, max_step=max_step, seed=seed)

    def _init_params(self, params_bound: ParamBound, max_step: int):
        """
        Initializes the parameters for the exponential scheduler.

        Args:
            params_bound (ParamBound): A dictionary specifying the parameter bounds.
            max_step (int): The maximum number of steps.

        Returns:
            None
        """
        self.params = {}
        for k, v in params_bound.items():
            assert len(v) == 2
            linear_space = np.linspace(0, np.log(10), max_step)
            exponential_space = np.exp(linear_space)
            exponential_space = (exponential_space - exponential_space[0]) / (
                exponential_space[-1] - exponential_space[0]
            )
            start_point = self.rng.uniform(v[0], v[1])
            direction = self.rng.choice([v[0], v[1]])
            if start_point < direction:
                self.params[k] = (
                    exponential_space * (direction - start_point) + start_point
                )
            else:
                self.params[k] = (
                    exponential_space[::-1] * (start_point - direction) + direction
                )


class LogarithmicScheduler(PrecomputedScheduler):
    """
    A scheduler that samples parameters logarithmically.
    Goes from the start point to the direction point in a logarithmic fashion.
    """

    def __init__(
        self,
        params_bound: ParamBound,
        max_step: int = 1000,
        seed: int | None = None,
    ) -> None:
        super().__init__(params_bound=params_bound, max_step=max_step, seed=seed)

    def _init_params(self, params_bound: ParamBound, max_step: int):
        """
        Initialize the parameters based on the given parameter bounds and maximum step.

        Args:
            params_bound (ParamBound): A dictionary containing the parameter bounds.
            max_step (int): The maximum step value.

        Returns:
            None
        """
        self.params = {}
        for k, v in params_bound.items():
            assert len(v) == 2
            linear_space = np.linspace(1, 10, max_step)
            logarithmic_space = np.log(linear_space)
            logarithmic_space = (
                logarithmic_space - logarithmic_space[0]
            ) / logarithmic_space[-1]
            start_point = self.rng.uniform(v[0], v[1])
            direction = self.rng.choice([v[0], v[1]])
            if start_point < direction:
                self.params[k] = (
                    logarithmic_space * (direction - start_point) + start_point
                )
            else:
                self.params[k] = (
                    logarithmic_space[::-1] * (start_point - direction) + direction
                )


class DynamicScheduler(Scheduler):
    """
    Abstract base class for dynamic schedulers.
    """

    def __init__(
        self,
        params_bound: ParamBound,
        radius: float,
        seed: Union[int, None] = None,
    ) -> None:
        self.radius = radius
        super().__init__(params_bound=params_bound, seed=seed)


class RandomScheduler(DynamicScheduler):
    """
    A scheduler that samples parameters updates randomly.
    """

    def __init__(
        self,
        params_bound: ParamBound,
        radius: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__(params_bound=params_bound, radius=radius, seed=seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is None:
            seed = np.random.randint(0, 1000)
        self.rng = np.random.default_rng(seed=seed)
        self.current_params = {
            k: self.rng.uniform(v[0], v[1]) for k, v in self.params_bound.items()
        }

    def sample(self, t: int) -> dict[str, float]:
        """
        Generate a sample of parameters by adding random noise to the current parameters.

        Args:
            t (int): The current time step.

        Returns:
            dict[str, float]: A dictionary containing the sampled parameters.
        """
        self.current_params = {
            k: self.current_params[k] + self.rng.uniform(-self.radius, self.radius)
            for k in self.params_bound.keys()
        }
        # clip to the bounds
        for k, v in self.params_bound.items():
            self.current_params[k] = np.clip(self.current_params[k], v[0], v[1])
        return self.current_params


class CosineScheduler(DynamicScheduler):
    """
    A scheduler that samples parameters updates using a cosine function.
    """

    def __init__(
        self,
        params_bound: ParamBound,
        radius: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__(params_bound=params_bound, radius=radius, seed=seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is None:
            seed = np.random.randint(0, 1000)
        self.rng = np.random.default_rng(seed=seed)
        self.phase_shift = {
            k: self.rng.uniform(0, 2 * np.pi) for k in self.params_bound.keys()
        }

    def sample(self, t: int) -> dict[str, float]:
        """
        Sample the current parameters based on the given time.

        Parameters:
            t (int): The time value.

        Returns:
            dict[str, float]: A dictionary containing the sampled parameters.
        """
        current_params = {
            k: (np.cos(t * self.radius * np.pi + self.phase_shift[k]) + 1)
            / 2
            * (v[1] - v[0])
            + v[0]
            for k, v in self.params_bound.items()
        }
        return current_params
