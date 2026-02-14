import itertools

import numpy as np
import torch
from gymnasium import spaces

from high_society.networks import build_mlp, build_discrete_mlp
from high_society.utils import get_device

device = get_device()


class Agent:

    def get_action(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        """Get an action based on the current observation."""
        raise NotImplementedError


class DiscreteAgent:
    """Base class for agents with discrete action spaces."""

    def get_action(self, observation: np.ndarray, action_mask: np.ndarray) -> tuple[int, float]:
        """Get an action based on the current observation and valid action mask.

        Args:
            observation: Flattened observation array
            action_mask: Boolean array where True = valid action

        Returns:
            Tuple of (action index, log_prob)
        """
        raise NotImplementedError


class RandomAgent(Agent):
    """Agent that makes random decisions during auctions.

    Strategy:
    - Randomly decides to bid or pass based on pass_probability
    - If bidding, samples a random raise_intensity

    Action is raise_intensity in [0, 1]:
    - 0 = pass
    - > 0 = raise (env computes actual bid)
    """

    def __init__(self, player_id: int, obs_space: spaces.Dict, pass_probability: float = 0.5, seed: int | None = None):
        """Initialize the random agent.

        Args:
            player_id: The player index this agent controls
            obs_space: The observation space (unused, kept for consistent interface)
            pass_probability: Probability of passing (0-1), default 0.5
            seed: Random seed for reproducibility
        """
        self.player_id = player_id
        self.pass_probability = pass_probability
        self.rng = np.random.RandomState(seed)

    def get_action(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        """Select an action based on the current observation.

        Args:
            observation: Flattened observation array

        Returns:
            Tuple of (raise_intensity action, log_prob placeholder)
        """
        # Randomly decide to pass
        if self.rng.random() < self.pass_probability:
            return np.array([0.0]), 0.0

        # Sample random raise_intensity in (0, 1]
        raise_intensity = self.rng.uniform(0.0, 1.0)
        return np.array([raise_intensity]), 0.0

class VanillaPGAgent(Agent):
    """An agent that uses the REINFORCE algorithm to learn a policy.

    Action is raise_intensity in [0, 1]:
    - 0 = pass
    - > 0 = raise (env computes actual bid)
    """

    def __init__(self, player_id: int, obs_space: spaces.Dict):
        self.player_id = player_id
        self.device = device
        obs_dim = sum(space.shape[0] for space in obs_space.spaces.values())
        self.obs_dim = obs_dim
        self.raise_beta_params_net = build_mlp(obs_dim, 2, 10, 256).to(self.device)
        parameters = itertools.chain(self.raise_beta_params_net.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=1e-3)


    @torch.no_grad
    def get_action(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        """Get an action based on the current observation.

        Returns:
            Tuple of (raise_intensity action, log_prob)
        """
        obs = torch.from_numpy(observation).to(self.device)
        beta_params = self.raise_beta_params_net(obs)
        assert beta_params.shape == (2,), f"Expected (2,), got {beta_params.shape}"
        alpha, beta = beta_params[0], beta_params[1]
        beta_distn = torch.distributions.Beta(alpha, beta)
        raise_intensity = beta_distn.rsample()
        log_prob = beta_distn.log_prob(raise_intensity)
        return np.array([float(raise_intensity)]), log_prob.item()

    def update(self, batch_traj_data: list[dict[str, np.ndarray]]):
        """Update the agent's policy based on a batch of trajectory data.

        Args:
            batch_traj_data: List of trajectory dicts, each containing
                'observations', 'actions', 'rewards' arrays from one game.
        """
        # Concatenate all trajectories from the batch 
        observations = torch.from_numpy(
            np.concatenate([traj["observations"] for traj in batch_traj_data], axis=0)
        ).float().to(self.device)
        actions = torch.from_numpy(
            np.concatenate([traj["actions"] for traj in batch_traj_data], axis=0)
        ).float().to(self.device)
        rewards = torch.from_numpy(
            np.concatenate([traj["rewards"] for traj in batch_traj_data], axis=0)
        ).float().to(self.device)

        # get the advantages
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # get the log_probs
        params = self.raise_beta_params_net(observations)
        alpha, beta = params[:, 0], params[:, 1]
        distns = torch.distributions.Beta(alpha, beta) # (n_samples, 1)
        log_probs = distns.log_prob(actions.squeeze(-1))

        self.optimizer.zero_grad()
        loss = -(log_probs * advantages).mean()
        loss.backward()
        self.optimizer.step()


class DiscreteRandomAgent(DiscreteAgent):
    """Agent that makes random decisions from valid actions.

    Uniformly samples from valid actions based on action mask.
    """

    def __init__(self, player_id: int, num_actions: int, seed: int | None = None):
        """Initialize the discrete random agent.

        Args:
            player_id: The player index this agent controls
            num_actions: Total number of possible actions
            seed: Random seed for reproducibility
        """
        self.player_id = player_id
        self.num_actions = num_actions
        self.rng = np.random.RandomState(seed)

    def get_action(self, observation: np.ndarray, action_mask: np.ndarray) -> tuple[int, float]:
        """Select a random valid action.

        Args:
            observation: Flattened observation array (unused, for interface consistency)
            action_mask: Boolean array where True = valid action

        Returns:
            Tuple of (action index, log_prob placeholder)
        """
        valid_actions = np.where(action_mask)[0]
        action = self.rng.choice(valid_actions)
        return int(action), 0.0


class DiscreteRandomPassAgent(DiscreteAgent):
    """Agent that makes random decisions from valid actions, including passing.

    Uniformly samples between passing and a random valid action based on action mask.
    """

    def __init__(self, player_id: int, pass_probability: float = 0.5, seed: int | None = None):
        """Initialize the discrete random pass agent.

        Args:
            player_id: The player index this agent controls
            pass_probability: Probability of passing (0-1), default 0.5
            seed: Random seed for reproducibility
        """
        self.player_id = player_id
        self.pass_probability = pass_probability
        self.rng = np.random.RandomState(seed)
        self.action_pass = 0

    def get_action(self, observation: np.ndarray, action_mask: np.ndarray) -> tuple[int, float]:
        """Select a random valid action.

        Args:
            observation: Flattened observation array (unused, for interface consistency)
            action_mask: Boolean array where True = valid action

        Returns:
            Tuple of (action index, log_prob placeholder)
        """

        valid_actions = np.where(action_mask)[0]
        if self.rng.random() < self.pass_probability:
            return self.action_pass, 0.0

        non_pass_actions = valid_actions[valid_actions != self.action_pass]
        if sum(non_pass_actions) == 0:
            return self.action_pass, 0.0
            
        action = self.rng.choice(non_pass_actions)
        return int(action), 0.0

class DQNAgent(DiscreteAgent):
    """Agent that uses a DQN to learn a policy.

    Action is action index.
    """

    def __init__(self, player_id: int, num_actions: int, obs_space: spaces.Dict, epsilon: float = 0.1):
        self.player_id = player_id
        self.num_actions = num_actions
        self.device = device
        self.epsilon = epsilon
        obs_dim = sum(space.shape[0] for space in obs_space.spaces.values())
        self.q_net = build_discrete_mlp(obs_dim, num_actions, 4, 64).to(self.device)
        self.target_q_net = build_discrete_mlp(obs_dim, num_actions, 4, 64).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.loss_fn = torch.nn.MSELoss()
        self.gamma = 0.99
        self.target_update_freq = 100
        self.current_step = 0

    @torch.no_grad()
    def get_action(self, observation: np.ndarray, action_mask: np.ndarray) -> tuple[int, float]:
        """Get an action based on the current observation with epsilon-greedy exploration.

        Returns:
            Tuple of (action index, log_prob placeholder)
        """
        valid_actions = np.where(action_mask)[0]

        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            action = np.random.choice(valid_actions)
            return int(action), 0.0

        obs = torch.from_numpy(observation).float().to(self.device)
        q_vals = self.q_net(obs)
        # Mask invalid actions with large negative value
        mask_tensor = torch.from_numpy(action_mask).float().to(self.device)
        q_vals = q_vals + (1 - mask_tensor) * -1e8
        action = q_vals.argmax(dim=-1)
        return int(action.item()), 0.0

    def update(self, batch_traj_data: list[dict[str, np.ndarray]]):
        """Update Q-network using batch of trajectory data."""
        observations = np.concatenate([traj["observations"] for traj in batch_traj_data], axis=0)
        actions = np.concatenate([traj["actions"] for traj in batch_traj_data], axis=0)
        action_masks = np.concatenate([traj["action_masks"] for traj in batch_traj_data], axis=0)
        rewards = np.concatenate([traj["rewards"] for traj in batch_traj_data], axis=0)
        terminateds = np.concatenate([traj["terminateds"] for traj in batch_traj_data], axis=0)

        # Build next observations/masks by rolling and zeroing out trajectory boundaries
        traj_lengths = np.array([len(traj["observations"]) for traj in batch_traj_data])
        boundary_indices = np.cumsum(traj_lengths) - 1  # last index of each trajectory

        next_observations = np.roll(observations, -1, axis=0)
        next_action_masks = np.roll(action_masks, -1, axis=0)
        next_observations[boundary_indices] = 0
        next_action_masks[boundary_indices] = 0

        # Convert to tensors
        observations = torch.from_numpy(observations).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        action_masks = torch.from_numpy(action_masks).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        terminateds = torch.from_numpy(terminateds).float().to(self.device)
        next_observations = torch.from_numpy(next_observations).float().to(self.device)
        next_action_masks = torch.from_numpy(next_action_masks).float().to(self.device)

        with torch.no_grad():
            target_next_q_values = self.target_q_net(next_observations)
            target_next_q_values = target_next_q_values + (1 - next_action_masks) * -1e8
            next_q_values = target_next_q_values.max(dim=1).values
            target_values = rewards + self.gamma * next_q_values * (1 - terminateds)

        cur_q_values = self.q_net(observations)
        cur_q_values = cur_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        self.optimizer.zero_grad()
        loss = self.loss_fn(cur_q_values, target_values)
        loss.backward()
        self.optimizer.step()

        self.current_step += 1
        if self.current_step % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())