import itertools

import numpy as np
import torch
from gymnasium import spaces

from high_society.networks import build_mlp


class Agent:

    def get_action(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        """Get an action based on the current observation."""
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
        obs_dim = sum(space.shape[0] for space in obs_space.spaces.values())
        self.raise_beta_params_net = build_mlp(obs_dim, 2, 2, 64)
        parameters = itertools.chain(self.raise_beta_params_net.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=1e-3)

    @torch.no_grad
    def get_action(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        """Get an action based on the current observation.

        Returns:
            Tuple of (raise_intensity action, log_prob)
        """
        obs = torch.from_numpy(observation)
        beta_params = self.raise_beta_params_net(obs)
        assert beta_params.shape == (2,), f"Expected (2,), got {beta_params.shape}"
        alpha, beta = beta_params[0], beta_params[1]
        beta_distn = torch.distributions.Beta(alpha, beta)
        raise_intensity = beta_distn.rsample()
        log_prob = beta_distn.log_prob(raise_intensity)
        return np.array([float(raise_intensity)]), log_prob.item()


        

    
