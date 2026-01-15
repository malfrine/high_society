from high_society.environment import SimpleHighSocietyEnv
from high_society.agents import VanillaPGAgent, RandomAgent
from high_society.utils import collect_trajectories


def test_collect_trajectories_basic():
    """Test that collect_trajectories returns correct structure and data types."""
    env = SimpleHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    obs_space = env.observation_space("player_0")
    agents = [
        VanillaPGAgent(player_id=0, obs_space=obs_space),
        RandomAgent(player_id=1, obs_space=obs_space, pass_probability=0.5, seed=43),
        RandomAgent(player_id=2, obs_space=obs_space, pass_probability=0.5, seed=44),
    ]

    result = collect_trajectories(env, agents, max_steps=10)

    # Should only collect data for VanillaPGAgent (player 0)
    assert 0 in result
    assert 1 not in result
    assert 2 not in result

    # Check all expected keys are present
    data = result[0]
    assert "observations" in data
    assert "actions" in data
    assert "log_probs" in data
    assert "rewards" in data
    assert "terminateds" in data
    assert "truncateds" in data

    # Check shapes are consistent
    T = len(data["log_probs"])
    assert T > 0, "Should have collected at least one step"
    assert data["observations"].shape[0] == T
    assert data["actions"].shape[0] == T
    assert data["rewards"].shape[0] == T
    assert data["terminateds"].shape[0] == T
    assert data["truncateds"].shape[0] == T


def test_collect_trajectories_truncation():
    """Test that truncation flag is set when max_steps is reached."""
    env = SimpleHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    obs_space = env.observation_space("player_0")
    agents = [
        VanillaPGAgent(player_id=0, obs_space=obs_space),
        RandomAgent(player_id=1, obs_space=obs_space, pass_probability=0.5, seed=43),
        RandomAgent(player_id=2, obs_space=obs_space, pass_probability=0.5, seed=44),
    ]

    # Use very small max_steps to ensure truncation
    result = collect_trajectories(env, agents, max_steps=5)

    data = result[0]
    # Last step should have truncated=True since we hit max_steps
    assert data["truncateds"][-1] == True


def test_collect_trajectories_multiple_trainable_agents():
    """Test collecting trajectories for multiple VanillaPGAgents."""
    env = SimpleHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    obs_space = env.observation_space("player_0")
    agents = [
        VanillaPGAgent(player_id=0, obs_space=obs_space),
        VanillaPGAgent(player_id=1, obs_space=obs_space),
        RandomAgent(player_id=2, obs_space=obs_space, pass_probability=0.5, seed=44),
    ]

    result = collect_trajectories(env, agents, max_steps=10)

    # Should collect for both VanillaPGAgents
    assert 0 in result
    assert 1 in result
    assert 2 not in result

    # Both should have data
    assert len(result[0]["log_probs"]) > 0
    assert len(result[1]["log_probs"]) > 0
