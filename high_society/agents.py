import numpy as np
from typing import Any
import torch
import torch.nn as nn
from high_society.networks import build_mlp


class RandomAgent:
    """Agent that makes random decisions during auctions.

    Strategy:
    - Randomly decides to bid or pass
    - If bidding, chooses a random valid bid amount within budget
    """

    def __init__(self, player_id: int, pass_probability: float = 0.5, seed: int | None = None):
        """Initialize the random agent.

        Args:
            player_id: The player index this agent controls
            pass_probability: Probability of passing (0-1), default 0.5
            seed: Random seed for reproducibility
        """
        self.player_id = player_id
        self.pass_probability = pass_probability
        self.rng = np.random.RandomState(seed)

    def get_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Select an action based on the current observation.

        Args:
            observation: Dictionary containing game state information

        Returns:
            Action array with bid amount (0 = pass)
        """
        remaining_money = float(observation["remaining_money"][0])
        current_bid = float(observation["current_round_bid"][0])
        min_bid = current_bid + 1

        # If we can't afford to bid, we must pass
        if remaining_money < min_bid:
            return np.array([0.0])

        # Randomly decide to pass
        if self.rng.random() < self.pass_probability:
            return np.array([0.0])

        # Make a random bid between min_bid and our remaining money
        max_bid = remaining_money
        bid_amount = self.rng.uniform(min_bid, max_bid)

        return np.array([bid_amount])

class VanillaPGAgent:
    """An agent that uses the REINFORCE algorithm to learn a policy."""

    def __init__(self, player_id: int, state_dim: int, action_dim: int):
        self.player_id = player_id
        self.mean_net = build_mlp(state_dim, action_dim, 2, 64)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        parameters = list(self.mean_net.parameters()) + [self.log_std]
        self.optimizer = torch.optim.Adam(parameters, lr=1e-3)

    
