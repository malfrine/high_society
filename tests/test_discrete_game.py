"""Tests for complete DiscreteHighSocietyEnv games"""
import pytest
import numpy as np
from high_society.environments.discrete import DiscreteHighSocietyEnv
from high_society.agents import DiscreteRandomAgent
from high_society.utils import cat_dict_array


def test_random_game_completes():
    """Test that a complete game with random agents finishes correctly."""
    env = DiscreteHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    agents = [
        DiscreteRandomAgent(player_id=0, num_actions=env.num_actions, seed=42),
        DiscreteRandomAgent(player_id=1, num_actions=env.num_actions, seed=43),
        DiscreteRandomAgent(player_id=2, num_actions=env.num_actions, seed=44),
    ]

    step_count = 0
    max_steps = 1000

    while not all(env.terminations.values()) and step_count < max_steps:
        agent_name = env.agent_selection
        agent_idx = env.agents.index(agent_name)
        agent = agents[agent_idx]

        obs = cat_dict_array(env.observe(agent_name))
        mask = env.get_action_mask(agent_name)
        action, _ = agent.get_action(obs, mask)

        env.step(action)
        step_count += 1

    assert step_count < max_steps, "Game did not complete within step limit"
    assert all(env.terminations.values()), "Not all agents terminated"
    assert env.game_state.remaining_special_cards == 0, "Game should end when 4 special cards drawn"

    # Get final state
    player_states = env.game_state.player_states
    rewards = env.rewards

    # Find minimum money (for elimination)
    min_money = min(player.total_money for player in player_states.values())

    # Verify at least one player has cards
    total_cards = sum(len(player.prestige_cards) for player in player_states.values())
    assert total_cards > 0, "No cards were won during the game"

    print(f"\nGame completed in {step_count} steps")
    print(f"Final state:")
    for i, agent_name in enumerate(env.agents):
        player_state = player_states[i]
        eliminated = "ELIMINATED" if player_state.total_money == min_money else ""
        print(f"  {agent_name}: Money={player_state.total_money}, "
              f"Prestige={player_state.total_prestige:.1f}, "
              f"Cards={len(player_state.prestige_cards)}, "
              f"Reward={rewards[agent_name]:.1f} {eliminated}")


def test_multiple_random_games():
    """Test that multiple games all complete successfully."""
    num_games = 20

    for game_num in range(num_games):
        env = DiscreteHighSocietyEnv(num_players=3)
        env.reset(seed=game_num)

        agents = [
            DiscreteRandomAgent(player_id=i, num_actions=env.num_actions, seed=game_num * 10 + i)
            for i in range(3)
        ]

        step_count = 0
        max_steps = 1000

        while not all(env.terminations.values()) and step_count < max_steps:
            agent_name = env.agent_selection
            agent_idx = env.agents.index(agent_name)
            agent = agents[agent_idx]

            obs = cat_dict_array(env.observe(agent_name))
            mask = env.get_action_mask(agent_name)
            action, _ = agent.get_action(obs, mask)

            env.step(action)
            step_count += 1

        assert step_count < max_steps, f"Game {game_num} did not complete"
        assert all(env.terminations.values()), f"Game {game_num} agents not all terminated"
        assert env.game_state.remaining_special_cards == 0, f"Game {game_num} did not end correctly"

    print(f"\nAll {num_games} games completed successfully")


def test_game_with_varying_player_counts():
    """Test that games work correctly with different numbers of players."""
    for num_players in [3, 4, 5]:
        env = DiscreteHighSocietyEnv(num_players=num_players)
        env.reset(seed=100)

        agents = [
            DiscreteRandomAgent(player_id=i, num_actions=env.num_actions, seed=100 + i)
            for i in range(num_players)
        ]

        step_count = 0
        max_steps = 2000

        while not all(env.terminations.values()) and step_count < max_steps:
            agent_name = env.agent_selection
            agent_idx = env.agents.index(agent_name)
            agent = agents[agent_idx]

            obs = cat_dict_array(env.observe(agent_name))
            mask = env.get_action_mask(agent_name)
            action, _ = agent.get_action(obs, mask)

            env.step(action)
            step_count += 1

        assert step_count < max_steps, f"Game with {num_players} players did not complete"
        assert all(env.terminations.values())
        assert env.game_state.remaining_special_cards == 0

        print(f"Game with {num_players} players completed in {step_count} steps")


def test_elimination_rule():
    """Test that the elimination rule works correctly."""
    env = DiscreteHighSocietyEnv(num_players=3)

    # Run multiple games and verify elimination logic
    for seed in range(10):
        env.reset(seed=seed)

        agents = [
            DiscreteRandomAgent(player_id=i, num_actions=env.num_actions, seed=seed * 10 + i)
            for i in range(3)
        ]

        step_count = 0
        while not all(env.terminations.values()) and step_count < 1000:
            agent_name = env.agent_selection
            agent_idx = env.agents.index(agent_name)
            agent = agents[agent_idx]

            obs = cat_dict_array(env.observe(agent_name))
            mask = env.get_action_mask(agent_name)
            action, _ = agent.get_action(obs, mask)

            env.step(action)
            step_count += 1

        # Verify elimination rule
        player_states = env.game_state.player_states
        min_money = min(p.total_money for p in player_states.values())

        # Count winners (reward = 1.0)
        winners = [name for name, reward in env.rewards.items() if reward == 1.0]

        # If everyone has the same (minimum) money, no winner
        all_same_money = all(p.total_money == min_money for p in player_states.values())
        if all_same_money:
            assert len(winners) == 0, "Should have no winner when all eliminated"
        else:
            # Winner should not have minimum money
            if len(winners) == 1:
                winner_idx = env.agents.index(winners[0])
                assert player_states[winner_idx].total_money > min_money, \
                    "Winner should not have minimum money"
