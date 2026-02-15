"""Tests for discrete trajectory collection"""
import numpy as np
from high_society.environments.discrete import DiscreteHighSocietyEnv, NUM_MONEY_CARDS
from high_society.agents import DiscreteRandomAgent
from high_society.main import collect_trajectories_discrete


def test_collect_trajectories_discrete_basic():
    """Test that collect_trajectories_discrete returns correct structure and data types."""
    env = DiscreteHighSocietyEnv(num_players=3)

    agents = [
        DiscreteRandomAgent(player_id=0, num_actions=env.num_actions, seed=42),
        DiscreteRandomAgent(player_id=1, num_actions=env.num_actions, seed=43),
        DiscreteRandomAgent(player_id=2, num_actions=env.num_actions, seed=44),
    ]

    # Only collect for player 0
    result = collect_trajectories_discrete(env, agents, max_steps=100)

    assert 0 in result

    data = result[0]
    assert "observations" in data
    assert "actions" in data
    assert "action_masks" in data
    assert "log_probs" in data
    assert "rewards" in data
    assert "terminateds" in data
    assert "truncateds" in data
    assert "won" in data

    # Check shapes are consistent
    T = len(data["log_probs"])
    assert T > 0, "Should have collected at least one step"
    assert data["observations"].shape[0] == T
    assert data["actions"].shape == (T,)
    assert data["action_masks"].shape == (T, env.num_actions)
    assert data["rewards"].shape == (T,)
    assert data["terminateds"].shape == (T,)
    assert data["truncateds"].shape == (T,)


def test_collect_trajectories_discrete_action_masks():
    """Test that action masks are correctly recorded."""
    env = DiscreteHighSocietyEnv(num_players=3)

    agents = [
        DiscreteRandomAgent(player_id=0, num_actions=env.num_actions, seed=42),
        DiscreteRandomAgent(player_id=1, num_actions=env.num_actions, seed=43),
        DiscreteRandomAgent(player_id=2, num_actions=env.num_actions, seed=44),
    ]

    result = collect_trajectories_discrete(env, agents, max_steps=100)
    data = result[0]

    # Action masks should be boolean arrays
    assert data["action_masks"].dtype == bool

    # Pass (action 0) should always be valid
    assert np.all(data["action_masks"][:, 0] == True)

    # Each action taken should have been valid at the time
    for t in range(len(data["actions"])):
        action = data["actions"][t]
        mask = data["action_masks"][t]
        assert mask[action], f"Action {action} at step {t} should have been valid"


def test_collect_trajectories_discrete_truncation():
    """Test that truncation flag is set when max_steps is reached."""
    env = DiscreteHighSocietyEnv(num_players=3)

    agents = [
        DiscreteRandomAgent(player_id=0, num_actions=env.num_actions, seed=42),
        DiscreteRandomAgent(player_id=1, num_actions=env.num_actions, seed=43),
        DiscreteRandomAgent(player_id=2, num_actions=env.num_actions, seed=44),
    ]

    # Use very small max_steps to ensure truncation
    result = collect_trajectories_discrete(env, agents, max_steps=5)

    data = result[0]
    assert data["truncateds"][-1] == True


def test_collect_trajectories_discrete_multiple_trainable():
    """Test collecting trajectories for multiple trainable agents."""
    env = DiscreteHighSocietyEnv(num_players=3)

    agents = [
        DiscreteRandomAgent(player_id=0, num_actions=env.num_actions, seed=42),
        DiscreteRandomAgent(player_id=1, num_actions=env.num_actions, seed=43),
        DiscreteRandomAgent(player_id=2, num_actions=env.num_actions, seed=44),
    ]

    result = collect_trajectories_discrete(env, agents, max_steps=100)

    assert 0 in result
    assert 1 in result

    assert len(result[0]["log_probs"]) > 0
    assert len(result[1]["log_probs"]) > 0


def test_collect_trajectories_discrete_rewards():
    """Test that rewards are correctly assigned."""
    env = DiscreteHighSocietyEnv(num_players=3)

    agents = [
        DiscreteRandomAgent(player_id=0, num_actions=env.num_actions, seed=42),
        DiscreteRandomAgent(player_id=1, num_actions=env.num_actions, seed=43),
        DiscreteRandomAgent(player_id=2, num_actions=env.num_actions, seed=44),
    ]

    result = collect_trajectories_discrete(env, agents, max_steps=1000)

    # Count winners
    winners = [pid for pid, data in result.items() if data["won"]]

    # Should have at most one winner
    assert len(winners) <= 1

    # If there's a winner, their rewards should all be 1.0
    for pid, data in result.items():
        if data["won"]:
            assert np.all(data["rewards"] == 1.0)
        else:
            assert np.all(data["rewards"] == -1.0)


def test_collect_trajectories_discrete_complete_game():
    """Test that a complete game can be collected."""
    env = DiscreteHighSocietyEnv(num_players=3)

    agents = [
        DiscreteRandomAgent(player_id=0, num_actions=env.num_actions, seed=42),
        DiscreteRandomAgent(player_id=1, num_actions=env.num_actions, seed=43),
        DiscreteRandomAgent(player_id=2, num_actions=env.num_actions, seed=44),
    ]

    result = collect_trajectories_discrete(env, agents, max_steps=1000)

    data = result[0]

    # Game should have completed (not truncated)
    assert data["truncateds"][-1] == False
    assert data["terminateds"][-1] == True
