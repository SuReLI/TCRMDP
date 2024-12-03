import copy
import itertools
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from .utils import (
    CriticNetwork,
    HatOmegaNetwork,
    PolicyNetwork,
    ReplayBuffer,
    Transition,
)


class M2TD3Config(TypedDict):
    policy_std_rate: float
    policy_noise_rate: float
    noise_clip_policy_rate: float
    omega_std_rate: list
    min_omega_std_rate: list
    noise_clip_omega_rate: float
    batch_size: int
    hatomega_num: int
    obs_dim: int
    action_dim: int
    replay_size: int
    critic_hidden_num: int
    critic_hidden_size: int
    p_bias: float
    p_lr: float
    restart_distance: bool
    restart_probability: bool
    policy_freq: int
    gamma: float
    polyak: float
    hatomega_parameter_distance: float
    minimum_prob: float
    ho_lr: float
    device: str
    seed: int
    dim_uncertainty_set: int


class M2TD3:
    """M2TD3 agent

    Parameters
    ----------
    config : Dict
        Configurations for the agent
    state_dim : int
        Number of state dimensions
    action_dim : int
        Number of action dimensions
    omega_dim : int
        Number of omega dimensions
    max_action : float
        Maximum value of action
    rand_state : np.random.RandomState
        Control random numbers
    min_omega : np.ndarray
        Minimum values for each omega dimension
    max_omega : np.ndarray
        Maximum values for each omega dimension
    policy_std_rate : float
        Rate for calculating the standard deviation of the policy noise
    policy_noise_rate : float
        Rate for calculating the magnitude of the policy noise
    noise_clip_policy_rate : float
        Rate for clipping the policy noise
    omega_std_rate : float
        Rate for calculating the standard deviation of the omega noise
    min_omega_std_rate : float
        Rate for calculating the minimum standard deviation of the omega noise
    max_steps : int
        Maximum number of steps
    batch_size : int
        Size of the mini-batch for training
    hatomega_num : int
        Number of hatomega functions
    replay_size : int
        Size of the replay buffer
    policy_hidden_num : int
        Number of hidden layers in the policy network
    policy_hidden_size : int
        Size of each hidden layer in the policy network
    critic_hidden_num : int
        Number of hidden layers in the critic network
    critic_hidden_size : int
        Size of each hidden layer in the critic network
    p_bias : np.ndarray
        Bias values for the critic network
    p_lr : float
        Learning rate for the policy network
    q_lr : float
        Learning rate for the critic network
    restart_distance : bool
        Flag indicating whether to restart hatomega functions based on distance
    restart_probability : bool
        Flag indicating whether to restart hatomega functions based on probability
    policy_freq : int
        Frequency of updating the policy network
    gamma : float
        Discount factor for future rewards
    polyak : float
        Polyak averaging coefficient for target networks
    hatomega_parameter_distance : float
        Distance threshold for restarting hatomega functions
    minimum_prob : float
        Minimum probability threshold for restarting hatomega functions
    hatomega_hidden_num : int
        Number of hidden layers in each hatomega function
    hatomega_hidden_size : int
        Size of each hidden layer in each hatomega function
    ho_lr : float
        Learning rate for the hatomega functions
    device : torch.device
        Device for running the agent

    Attributes
    ----------
    config : Dict
        Configurations for the agent
    device : torch.device
        Device for running the agent
    rand_state : np.random.RandomState
        Control random numbers
    state_dim : int
        Number of state dimensions
    action_dim : int
        Number of action dimensions
    omega_dim : int
        Number of omega dimensions
    min_omega : np.ndarray
        Minimum values for each omega dimension
    max_omega : np.ndarray
        Maximum values for each omega dimension
    min_omega_tensor : torch.Tensor
        Minimum values for each omega dimension as a tensor
    max_omega_tensor : torch.Tensor
        Maximum values for each omega dimension as a tensor
    max_action : float
        Maximum value of action
    policy_std_rate : float
        Rate for calculating the standard deviation of the policy noise
    policy_noise_rate : float
        Rate for calculating the magnitude of the policy noise
    noise_clip_policy_rate : float
        Rate for clipping the policy noise
    omega_std_rate : float
        Rate for calculating the standard deviation of the omega noise
    min_omega_std_rate : float
        Rate for calculating the minimum standard deviation of the omega noise
    max_steps : int
        Maximum number of steps
    batch_size : int
        Size of the mini-batch for training
    hatomega_num : int
        Number of hatomega functions
    replay_size : int
        Size of the replay buffer
    policy_hidden_num : int
        Number of hidden layers in the policy network
    policy_hidden_size : int
        Size of each hidden layer in the policy network
    critic_hidden_num : int
        Number of hidden layers in the critic network
    critic_hidden_size : int
        Size of each hidden layer in the critic network
    p_bias : np.ndarray
        Bias values for the critic network
    p_lr : float
        Learning rate for the policy network
    q_lr : float
        Learning rate for the critic network
    restart_distance : bool
        Flag indicating whether to restart hatomega functions based on distance
    restart_probability : bool
        Flag indicating whether to restart hatomega functions based on probability
    policy_freq : int
        Frequency of updating the policy network
    gamma : float
        Discount factor for future rewards
    polyak : float
        Polyak averaging coefficient for target networks
    hatomega_parameter_distance : float
        Distance threshold for restarting hatomega functions
    minimum_prob : float
        Minimum probability threshold for restarting hatomega functions
    hatomega_hidden_num : int
        Number of hidden layers in each hatomega function
    hatomega_hidden_size : int
        Size of each hidden layer in each hatomega function
    ho_lr : float
        Learning rate for the hatomega functions
    policy_std : float
        Standard deviation of the policy noise
    policy_noise : float
        Magnitude of the policy noise
    noise_clip_policy : float
        Clipping threshold for the policy noise
    omega_std : list
        Standard deviation of the omega noise for each dimension
    min_omega_std : list
        Minimum standard deviation of the omega noise for each dimension
    omega_std_step : np.ndarray
        Step size for adjusting the standard deviation of the omega noise
    omega_noise : float
        Magnitude of the omega noise
    noise_clip_omega : torch.Tensor
        Clipping threshold for the omega noise
    hatomega_input : torch.Tensor
        Input tensor for the hatomega functions
    hatomega_input_batch : torch.Tensor
        Batch input tensor for the hatomega functions
    hatomega_prob : list
        Probability distribution for selecting hatomega functions
    element_list : list
        List of indices for hatomega functions
    update_omega : list
        List of flags indicating whether to update each omega dimension
    policy_network : PolicyNetwork
        Policy network for selecting actions
    critic_network : CriticNetwork
        Critic network for estimating Q-values
    hatomega_list : list
        List of hatomega functions
    optimizer_hatomega_list : list
        List of optimizers for the hatomega functions
    policy_target : PolicyNetwork
        Target policy network for stability during training
    critic_target : CriticNetwork
        Target critic network for stability during training
    optimizer_policy : torch.optim.Adam
        Optimizer for the policy network
    optimizer_critic_p : torch.optim.Adam
        Optimizer for the critic network
    replay_buffer : ReplayBuffer
        Replay buffer for storing and sampling transitions

    Methods
    -------
    _init_network(state_dim, action_dim, omega_dim, max_action, config)
        Initialize the policy and critic networks
    _init_optimizer(config)
        Initialize the optimizers
    add_memory(*args)
        Add a transition to the replay buffer
    get_action(state, greedy=False)
        Get an action from the policy network
    get_omega(greedy=False)
        Get an omega value from the hatomega functions
    _buffer2batch()
        Create a mini-batch from the replay buffer
    train(step)
        Train the agent for one step
    _update_critic(state_batch, action_batch, next_state_batch, reward_batch, not_done_batch, omega_batch)
        Update the critic network
    _update_actor(state_batch)
        Update the policy network
    _update_target()
        Update the target networks
    _calc_diff()
        Calculate the difference between the current omega values and the hatomega parameters
    _minimum_prob()
        Find the hatomega functions with minimum probabilities
    _select_hatomega()
        Select a hatomega function based on the probability distribution
    _init_hatomega(index)
        Initialize a hatomega function
    _init_hatomega_prob(index)
        Initialize the probability of a hatomega function

    """

    def __init__(
        self,
        state_dim,
        action_dim,
        omega_dim,
        max_action,
        device,
        min_omega: np.ndarray,
        max_omega: np.ndarray,
        rand_state: int = 0,
        policy_std_rate: float = 0.1,
        policy_noise_rate: float = 0.2,
        noise_clip_policy_rate: float = 0.5,
        noise_clip_omega_rate: float = 0.5,
        omega_std_rate: float = 1.0,
        min_omega_std_rate: float = 0.1,
        max_steps: int = 2e6,
        batch_size: int = 100,
        hatomega_num: int = 5,
        replay_size: int = 1e6,
        policy_hidden_num: int = 1,
        policy_hidden_size: int = 256,
        critic_hidden_num: int = 1,
        critic_hidden_size: int = 256,
        p_bias: np.ndarray = 0,
        p_lr: float = 3e-4,
        q_lr: float = 3e-4,
        restart_distance: bool = True,
        restart_probability: bool = True,
        policy_freq: int = 2,
        gamma: float = 0.99,
        polyak: float = 5e-3,
        hatomega_parameter_distance: float = 0.1,
        minimum_prob: float = 5e-2,
        hatomega_hidden_num: int = 0,
        hatomega_hidden_size: int = 256,
        ho_lr: float = 3e-4,
    ):
        self.device = device
        self.rand_state = rand_state

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.omega_dim = omega_dim
        self.min_omega = min_omega
        self.max_omega = max_omega
        self.min_omega_tensor = torch.tensor(
            min_omega, dtype=torch.float, device=device
        )
        self.max_omega_tensor = torch.tensor(
            max_omega, dtype=torch.float, device=device
        )
        self.max_action = max_action
        self.policy_std_rate = policy_std_rate
        self.policy_noise_rate = policy_noise_rate
        self.noise_clip_policy_rate = noise_clip_policy_rate
        self.noise_clip_omega_rate = noise_clip_omega_rate
        self.omega_std_rate = omega_std_rate
        self.min_omega_std_rate = min_omega_std_rate
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.hatomega_num = hatomega_num
        self.replay_size = replay_size
        self.policy_hidden_num = policy_hidden_num
        self.policy_hidden_size = policy_hidden_size
        self.critic_hidden_num = critic_hidden_num
        self.critic_hidden_size = critic_hidden_size
        self.p_bias = p_bias
        self.p_lr = p_lr
        self.q_lr = q_lr
        self.restart_distance = restart_distance
        self.restart_probability = restart_probability
        self.policy_freq = policy_freq
        self.gamma = gamma
        self.polyak = polyak
        self.hatomega_parameter_distance = hatomega_parameter_distance
        self.minimum_prob = minimum_prob
        self.hatomega_hidden_num = hatomega_hidden_num
        self.hatomega_hidden_size = hatomega_hidden_size
        self.ho_lr = ho_lr

        self.policy_std = self.policy_std_rate * max_action
        self.policy_noise = self.policy_noise_rate * max_action
        self.noise_clip_policy = self.noise_clip_policy_rate * max_action

        self.omega_std = list(
            self.omega_std_rate * (self.max_omega - self.min_omega) / 2
        )
        self.min_omega_std = list(
            self.min_omega_std_rate * (self.max_omega - self.min_omega) / 2
        )
        self.omega_std_step = (
            np.array(self.omega_std) - np.array(self.min_omega_std)
        ) / self.max_steps
        self.omega_noise = (
            self.policy_noise_rate * (self.max_omega - self.min_omega) / 2
        )
        self.noise_clip_omega = torch.tensor(
            self.noise_clip_omega_rate * (self.max_omega - self.min_omega) / 2,
            device=self.device,
            dtype=torch.float,
        )

        self.hatomega_input = torch.tensor([[1]], dtype=torch.float, device=self.device)
        self.hatomega_input_batch = torch.tensor(
            [[1] * self.batch_size],
            dtype=torch.float,
            device=self.device,
        ).view(self.batch_size, 1)
        self.hatomega_prob = [1 / self.hatomega_num for _ in range(self.hatomega_num)]
        self.element_list = [i for i in range(self.hatomega_num)]
        self.update_omega = [0 for _ in range(len(self.max_omega))]

        self._init_network(state_dim, action_dim, omega_dim, max_action)
        self._init_optimizer()

        self.replay_buffer = ReplayBuffer(rand_state, capacity=self.replay_size)

    def _init_network(self, state_dim, action_dim, omega_dim, max_action):
        """Initialize network

        Parameters
        ----------
        state_dim : int
            Number of state dimensions
        action_dim : int
            Number of action dimensions
        omega_dim : int
            Number of omega dimensions
        max_action : float
            Maximum value of action
        config : Dict
            configs

        """

        self.policy_network = PolicyNetwork(
            state_dim,
            action_dim,
            self.policy_hidden_num,
            self.policy_hidden_size,
            max_action,
            self.device,
        ).to(self.device)

        self.critic_network = CriticNetwork(
            state_dim,
            action_dim,
            omega_dim,
            self.critic_hidden_num,
            self.critic_hidden_size,
            self.p_bias,
        ).to(self.device)

        self.hatomega_list = [None for _ in range(self.hatomega_num)]
        self.optimizer_hatomega_list = [None for _ in range(self.hatomega_num)]
        for i in range(self.hatomega_num):
            self._init_hatomega(i)

        self.policy_target = copy.deepcopy(self.policy_network)
        self.critic_target = copy.deepcopy(self.critic_network)

    def _init_optimizer(self):
        """Initialize optimizer

        Parameters
        ----------
        config : Dict
            configs

        """

        self.optimizer_policy = optim.Adam(
            self.policy_network.parameters(), lr=self.p_lr
        )
        self.optimizer_critic_p = optim.Adam(
            self.critic_network.parameters(), lr=self.q_lr
        )

    def add_memory(self, *args):
        """Add transition to replay buffer

        Parameters
        ----------
        args :
            "state", "action", "next_state", "reward", "done", "omega"

        """

        transition = Transition(*args)
        self.replay_buffer.append(transition)

    def get_action(self, state, greedy=False):
        """Get action

        Parameters
        ----------
        state : np.Array
            state
        greedy : bool
            flag whether or not to perform greedy behavior

        """

        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).view(
            -1, self.state_dim
        )
        action = self.policy_network(state_tensor)
        if not greedy:
            noise = torch.tensor(
                self.rand_state.normal(0, self.policy_std),
                dtype=torch.float,
                device=self.device,
            )
            action = action + noise
        return action.squeeze(0).detach().cpu().numpy()

    def get_omega(self, greedy=False):
        """Get omega

        Parameters
        ----------
        greedy : bool
            flag whether or not to perform greedy behavior

        """

        dis_restart_flag = False
        prob_restart_flag = False
        if self.restart_distance:
            change_hatomega_index_list = self._calc_diff()
            for index in change_hatomega_index_list:
                self._init_hatomega(index)
                self._init_hatomega_prob(index)
                dis_restart_flag = True
        if self.restart_probability:
            change_hatomega_index_list = self._minimum_prob()
            for index in change_hatomega_index_list:
                self._init_hatomega(index)
                self._init_hatomega_prob(index)
                prob_restart_flag = True

        hatomega_index = self._select_hatomega()
        omega = self.hatomega_list[hatomega_index](self.hatomega_input)

        if not greedy:
            noise = torch.tensor(
                self.rand_state.normal(0, self.omega_std),
                dtype=torch.float,
                device=self.device,
            )
            omega += noise
        return (
            omega.squeeze(0).detach().cpu().numpy(),
            dis_restart_flag,
            prob_restart_flag,
        )

    def _buffer2batch(self):
        """Make mini-batch from buffer datas"""

        transitions = self.replay_buffer.sample(self.batch_size)
        if transitions is None:
            return None, None, None, None, None, None
        batch = Transition(*zip(*transitions))
        state_batch = torch.tensor(
            np.array(batch.state, dtype=float), device=self.device, dtype=torch.float
        )
        action_batch = torch.tensor(
            np.array(batch.action, dtype=float), device=self.device, dtype=torch.float
        )
        next_state_batch = torch.tensor(
            np.array(batch.next_state, dtype=float),
            device=self.device,
            dtype=torch.float,
        )
        reward_batch = torch.tensor(
            np.array(batch.reward, dtype=float), device=self.device, dtype=torch.float
        ).unsqueeze(1)
        not_done = np.array([(not don) for don in batch.done])
        not_done_batch = torch.tensor(
            np.array(not_done), device=self.device, dtype=torch.float
        ).unsqueeze(1)
        omega_batch = torch.tensor(
            np.array(batch.omega, dtype=float), device=self.device, dtype=torch.float
        )
        return (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            not_done_batch,
            omega_batch,
        )

    def train(self, step):
        """Training one step

        Parameters
        ----------
        step : int
            Number of current step

        """

        (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            not_done_batch,
            omega_batch,
        ) = self._buffer2batch()
        if state_batch is None:
            return None, None, None

        self._update_critic(
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            not_done_batch,
            omega_batch,
        )
        if step % self.policy_freq == 0:
            self._update_actor(state_batch)

            self._update_target()

    def _update_critic(
        self,
        state_batch,
        action_batch,
        next_state_batch,
        reward_batch,
        not_done_batch,
        omega_batch,
    ):
        """Update critic network

        Parameters
        ----------
        state_batch : torch.Tensor
            state batch
        action_batch : torch.Tensor
            action batch
        next_state_batch : torch.Tensor
            next state batch
        reward_batch : torch.Tensor
            reward batch
        not_done_batch : torch.Tensor
            not done batch
        omega_batch : torch.Tensor
            omega batch

        """

        with torch.no_grad():
            action_noise = (torch.randn_like(action_batch) * self.policy_noise).clamp(
                -self.noise_clip_policy, self.noise_clip_policy
            )
            next_action_batch = (
                self.policy_target(next_state_batch) + action_noise
            ).clamp(-self.max_action, self.max_action)
            omega_noise = torch.max(
                torch.min(
                    (
                        torch.randn_like(omega_batch)
                        * torch.tensor(
                            self.omega_noise, device=self.device, dtype=torch.float
                        )
                    ),
                    self.noise_clip_omega,
                ),
                -self.noise_clip_omega,
            )
            next_omega_batch = torch.max(
                torch.min((omega_batch + omega_noise), self.max_omega_tensor),
                self.min_omega_tensor,
            )

            targetQ1, targetQ2 = self.critic_target(
                next_state_batch, next_action_batch, next_omega_batch
            )
            targetQ = torch.min(targetQ1, targetQ2)
            targetQ = reward_batch + not_done_batch * self.gamma * targetQ

        currentQ1, currentQ2 = self.critic_network(
            state_batch, action_batch, omega_batch
        )
        critic_loss = F.mse_loss(currentQ1, targetQ) + F.mse_loss(currentQ2, targetQ)

        self.optimizer_critic_p.zero_grad()
        critic_loss.backward()
        self.optimizer_critic_p.step()

    def _update_actor(self, state_batch):
        """Update actor network

        Parameters
        ----------
        state_batch : torch.Tensor
            state batch
        """

        worst_policy_loss_index = None
        worst_policy_loss = None
        worst_policy_loss_value = -np.inf
        for hatomega_index in range(self.hatomega_num):
            hatomega_batch = self.hatomega_list[hatomega_index](
                self.hatomega_input_batch
            )

            policy_loss = -self.critic_network.Q1(
                state_batch, self.policy_network(state_batch), hatomega_batch.detach()
            ).mean()
            if policy_loss.item() >= worst_policy_loss_value:
                self.update_omega = list(
                    self.hatomega_list[hatomega_index](self.hatomega_input)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                worst_policy_loss = policy_loss
                worst_policy_loss_index = hatomega_index
                worst_policy_loss_value = policy_loss.item()

        hatomega_batch = self.hatomega_list[worst_policy_loss_index](
            self.hatomega_input_batch
        )

        hatomega_loss = self.critic_network.Q1(
            state_batch, self.policy_network(state_batch).detach(), hatomega_batch
        ).mean()
        self.optimizer_hatomega_list[worst_policy_loss_index].zero_grad()
        hatomega_loss.backward()
        self.optimizer_hatomega_list[worst_policy_loss_index].step()

        self.optimizer_policy.zero_grad()
        worst_policy_loss.backward()
        self.optimizer_policy.step()

        self._update_hatomega_prob(worst_policy_loss_index)

    def _update_target(self):
        """Update target network"""

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic_network.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.polyak) + param.data * self.polyak
            )

        for target_param, param in zip(
            self.policy_target.parameters(), self.policy_network.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.polyak) + param.data * self.polyak
            )

    def _calc_diff(self):
        """Compute the distance between hatomegas"""

        change_hatomega_index_list = []
        hatomega_parameter_list = []
        for i in range(self.hatomega_num):
            hatomega_parameter_list.append(
                self.hatomega_list[i](self.hatomega_input)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
        for hatomega_pair in itertools.combinations(hatomega_parameter_list, 2):
            distance = np.linalg.norm(
                np.array(hatomega_pair[0]) - np.array(hatomega_pair[1]), ord=1
            )
            if distance <= self.hatomega_parameter_distance:
                change_hatomega_index_list.append(
                    hatomega_parameter_list.index(hatomega_pair[0])
                )
        return change_hatomega_index_list

    def _update_hatomega_prob(self, index):
        """Update selection probability for hatomega

        Parameters
        ----------
        index : int
            Index of hatomega to be updated

        """

        p = [0] * self.hatomega_num
        p[index] = 1
        coeff = 1 / self.current_episode_len
        for i in range(self.hatomega_num):
            self.hatomega_prob[i] = self.hatomega_prob[i] * (1 - coeff) + coeff * p[i]

    def _minimum_prob(self):
        """Extract the index of the hatomega that has a selection probability below a threshold value."""

        indexes = []
        for index in range(self.hatomega_num):
            prob = self.hatomega_prob[index]
            if prob < self.minimum_prob:
                indexes.append(index)
        return indexes

    def _init_hatomega(self, index):
        """Initialize hatomega

        Parameters
        ----------
        index : int
            Index of hatomega to initialize
        """

        hatomega = HatOmegaNetwork(
            self.omega_dim,
            self.min_omega,
            self.max_omega,
            self.hatomega_hidden_num,
            self.hatomega_hidden_size,
            self.rand_state,
            self.device,
        ).to(self.device)
        optimizer_hatomega = optim.Adam(hatomega.parameters(), lr=self.ho_lr)
        self.hatomega_list[index] = hatomega
        self.optimizer_hatomega_list[index] = optimizer_hatomega

    def _init_hatomega_prob(self, index):
        """Initialize selection probability for hatomega

        Parameters
        ----------
        index : int
            Index of hatomega to initialize
        """

        self.hatomega_prob[index] = 0
        sum_prob = sum(self.hatomega_prob)
        p = sum_prob / (self.hatomega_num - 1)
        self.hatomega_prob[index] = p

    def _select_hatomega(self):
        """Select hatomega"""

        self.hatomega_prob = list(
            np.array(self.hatomega_prob) / sum(self.hatomega_prob)
        )
        select_index = self.rand_state.choice(
            a=self.element_list, size=1, p=self.hatomega_prob
        )
        return select_index[0]

    def _update_omega_std(self):
        """Update omega std"""

        if self.omega_std >= self.min_omega_std:
            self.omega_std = list(np.array(self.omega_std) - self.omega_std_step)

    def set_current_episode_len(self, episode_len):
        """Set length of episode

        Parameters
        ----------
        episode_len : int
            Length of current episode.
        """
        self.current_episode_len = episode_len

    def get_qvalue_list(self):
        """Retrieve the Q value for each hatomega"""

        qvalue_list = []
        transitions = self.replay_buffer.sample(self.batch_size)
        for hatomega_index in range(self.hatomega_num):
            if transitions is None:
                qvalue_list.append(0)
                continue
            batch = Transition(*zip(*transitions))
            state_batch = torch.tensor(
                batch.state, device=self.device, dtype=torch.float
            )
            q_value = self.critic_network.Q1(
                state_batch,
                self.policy_network(state_batch),
                self.hatomega_list[hatomega_index](self.hatomega_input_batch),
            ).mean()
            qvalue_list.append(q_value.item())
        return qvalue_list
