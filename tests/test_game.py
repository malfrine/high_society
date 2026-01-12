"""Tests for complete High Society games"""
import pytest
import numpy as np
from high_society.environment import HighSocietyEnvSimple
from high_society.agents import RandomAgent


def test_random_game_completes():
    """Test that a complete game with random agents finishes correctly.

    Verifies:
    - Game runs to completion without errors
    - Game ends when 4 special cards are drawn
    - All agents are terminated
    - Rewards are assigned correctly (eliminated players get 0)
    - Winner has highest prestige among non-eliminated players
    """
    env = HighSocietyEnvSimple(name="test", num_players=3)
    env.reset(seed=42)

    # Create random agents with different strategies
    agents = [
        RandomAgent(player_id=0, pass_probability=0.3, seed=42),  # Aggressive
        RandomAgent(player_id=1, pass_probability=0.5, seed=43),  # Balanced
        RandomAgent(player_id=2, pass_probability=0.7, seed=44),  # Conservative
    ]

    step_count = 0
    max_steps = 1000  # Safety limit to prevent infinite loops

    # Run game to completion
    while not all(env.terminations.values()) and step_count < max_steps:
        agent_name = env.agent_selection
        agent_idx = env.agents.index(agent_name)
        agent = agents[agent_idx]

        obs = env.observe(agent_name)
        action = agent.get_action(obs)

        env.step(action)
        step_count += 1

    # Verify game completed successfully
    assert step_count < max_steps, "Game did not complete within step limit"
    assert all(env.terminations.values()), "Not all agents terminated"

    # Verify game ended correctly (4 special cards drawn)
    assert env.game_state.remaining_special_cards == 0, "Game should end when 4 special cards drawn"

    # Get final state
    player_states = env.game_state.player_states
    rewards = env.rewards

    # Find minimum money (for elimination)
    min_money = min(player.total_money for player in player_states.values())

    # Verify elimination rule is applied correctly
    for i, agent_name in enumerate(env.agents):
        player_state = player_states[i]
        reward = rewards[agent_name]

        if player_state.total_money == min_money:
            # Eliminated player should have 0 reward
            assert reward == 0, f"Player {i} eliminated but got reward {reward}"
        else:
            # Non-eliminated player reward should equal their prestige
            assert reward == player_state.total_prestige, \
                f"Player {i} reward {reward} != prestige {player_state.total_prestige}"

    # Verify at least one player has cards (someone bid on something)
    total_cards = sum(len(player.prestige_cards) for player in player_states.values())
    assert total_cards > 0, "No cards were won during the game"

    print(f"\nGame completed in {step_count} steps")
    print(f"Final state:")
    for i, agent_name in enumerate(env.agents):
        player_state = player_states[i]
        print(f"  {agent_name}: Money={player_state.total_money:.1f}, "
              f"Prestige={player_state.total_prestige:.1f}, "
              f"Cards={len(player_state.prestige_cards)}, "
              f"Reward={rewards[agent_name]:.1f}")


def test_multiple_random_games():
    """Test that multiple games with random agents all complete successfully.

    This is a stress test to ensure the environment is robust.
    """
    num_games = 10

    for game_num in range(num_games):
        env = HighSocietyEnvSimple(name=f"test_{game_num}", num_players=3)
        env.reset(seed=game_num)

        agents = [
            RandomAgent(player_id=i, pass_probability=0.5, seed=game_num * 10 + i)
            for i in range(3)
        ]

        step_count = 0
        max_steps = 1000

        while not all(env.terminations.values()) and step_count < max_steps:
            agent_name = env.agent_selection
            agent_idx = env.agents.index(agent_name)
            agent = agents[agent_idx]

            obs = env.observe(agent_name)
            action = agent.get_action(obs)

            env.step(action)
            step_count += 1

        # Verify each game completed successfully
        assert step_count < max_steps, f"Game {game_num} did not complete"
        assert all(env.terminations.values()), f"Game {game_num} agents not all terminated"
        assert env.game_state.remaining_special_cards == 0, \
            f"Game {game_num} did not end at correct time"

    print(f"\nAll {num_games} games completed successfully")


def test_game_with_varying_player_counts():
    """Test that games work correctly with different numbers of players."""
    for num_players in [3, 4, 5]:
        env = HighSocietyEnvSimple(name=f"test_{num_players}p", num_players=num_players)
        env.reset(seed=100)

        agents = [
            RandomAgent(player_id=i, pass_probability=0.5, seed=100 + i)
            for i in range(num_players)
        ]

        step_count = 0
        max_steps = 2000  # More steps for more players

        while not all(env.terminations.values()) and step_count < max_steps:
            agent_name = env.agent_selection
            agent_idx = env.agents.index(agent_name)
            agent = agents[agent_idx]

            obs = env.observe(agent_name)
            action = agent.get_action(obs)

            env.step(action)
            step_count += 1

        assert step_count < max_steps, f"Game with {num_players} players did not complete"
        assert all(env.terminations.values())
        assert env.game_state.remaining_special_cards == 0

        print(f"Game with {num_players} players completed in {step_count} steps")
