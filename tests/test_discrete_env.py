"""Tests for DiscreteHighSocietyEnv"""
import pytest
import numpy as np
from high_society.environments.discrete import DiscreteHighSocietyEnv, MONEY_CARD_VALUES, ACTION_PASS
from high_society.agents import DiscreteRandomAgent
from high_society.utils import cat_dict_array


def test_env_initialization():
    """Test that environment initializes correctly."""
    env = DiscreteHighSocietyEnv(num_players=3)

    assert env.num_players == 3
    assert len(env.agents) == 3
    assert env.num_actions == 11  # PASS + 10 cards


def test_action_mask_initial_state():
    """Test that action mask is correct at game start."""
    env = DiscreteHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    mask = env.get_action_mask(env.agent_selection)

    # Can always pass
    assert mask[ACTION_PASS] == True

    # At start, current bid is 0, so any card can beat it
    # All cards should be valid (player has all cards 1-10)
    for card_value in MONEY_CARD_VALUES:
        assert mask[card_value] == True, f"Card {card_value} should be valid at start"


def test_action_mask_after_bid():
    """Test that action mask updates correctly after a bid."""
    env = DiscreteHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    # Player 0 bids card 5
    assert env.agent_selection == "player_0"
    env.step(5)  # Add card 5, bid is now 5

    # Player 1's turn - needs to beat 5
    assert env.agent_selection == "player_1"
    mask = env.get_action_mask("player_1")

    # Pass is always valid
    assert mask[ACTION_PASS] == True

    # Cards 1-5 cannot beat bid of 5 (need > 5)
    for card_value in range(1, 6):
        assert mask[card_value] == False, f"Card {card_value} should not beat bid of 5"

    # Cards 6-10 can beat bid of 5
    for card_value in range(6, 11):
        assert mask[card_value] == True, f"Card {card_value} should beat bid of 5"


def test_action_mask_card_in_bid():
    """Test that cards already in bid are masked out."""
    env = DiscreteHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    # Player 0 bids card 5
    env.step(5)
    # Player 1 bids card 6
    env.step(6)
    # Player 2 bids card 7
    env.step(7)

    # Back to player 0 - needs to beat 7, and card 5 is already in their bid
    assert env.agent_selection == "player_0"
    mask = env.get_action_mask("player_0")

    # Card 5 is in bid, should be masked
    assert mask[5] == False, "Card 5 is in bid, should be masked"

    # Player 0's current bid is 5, so cards that would make total > 7:
    # Need (5 + card) > 7, so card > 2
    # But card 5 is already in bid
    # Valid: 3, 4, 6, 7, 8, 9, 10 (cards > 2 that aren't 5)
    for card_value in [3, 4, 6, 7, 8, 9, 10]:
        assert mask[card_value] == True, f"Card {card_value} should be valid"

    for card_value in [1, 2]:
        assert mask[card_value] == False, f"Card {card_value} should not beat bid"


def test_pass_returns_cards():
    """Test that passing returns bid cards to player's hand."""
    env = DiscreteHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    # Player 0 bids card 5
    env.step(5)
    # Player 1 bids card 6 (must beat 5)
    env.step(6)
    # Player 2 bids card 7 (must beat 6)
    env.step(7)

    # Player 0's turn again - they pass
    assert env.agent_selection == "player_0"
    env.step(ACTION_PASS)

    # Player 1 passes
    env.step(ACTION_PASS)

    # Player 2 wins the auction
    # Player 0 should have card 5 back (they passed, so cards returned)
    player_0_state = env.game_state.player_states[0]
    assert any(mc.value == 5 for mc in player_0_state.money_cards), \
        "Player 0 should have card 5 back after passing"

    # Player 2 should have lost card 7 (they won)
    player_2_state = env.game_state.player_states[2]
    assert not any(mc.value == 7 for mc in player_2_state.money_cards), \
        "Player 2 should have lost card 7 after winning"


def test_winner_loses_cards():
    """Test that auction winner permanently loses bid cards."""
    env = DiscreteHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    # Player 0 bids card 5
    env.step(5)
    # Player 1 passes
    env.step(ACTION_PASS)
    # Player 2 passes
    env.step(ACTION_PASS)

    # Player 0 wins - should lose card 5
    player_0_state = env.game_state.player_states[0]
    assert not any(mc.value == 5 for mc in player_0_state.money_cards), \
        "Player 0 should have lost card 5 after winning"
    assert player_0_state.total_money == 55 - 5, \
        "Player 0's total money should be reduced by 5"


def test_random_game_completes():
    """Test that a complete game with random agents finishes correctly."""
    env = DiscreteHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    agents = [
        DiscreteRandomAgent(player_id=i, num_actions=env.num_actions, seed=42 + i)
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

    assert step_count < max_steps, "Game did not complete within step limit"
    assert all(env.terminations.values()), "Not all agents terminated"
    assert env.game_state.remaining_special_cards == 0, "Game should end when 4 special cards drawn"

    # Verify rewards assigned
    rewards = list(env.rewards.values())
    assert 1.0 in rewards or all(r == -1.0 for r in rewards), \
        "Should have exactly one winner or all eliminated"


def test_multiple_random_games():
    """Test that multiple games with random agents all complete."""
    num_games = 10

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


def test_observation_shape():
    """Test that observations have correct shape."""
    env = DiscreteHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    obs = env.observe(env.agent_selection)

    assert obs["total_prestige"].shape == (1,)
    assert obs["remaining_special_cards"].shape == (1,)
    assert obs["is_last_round"].shape == (1,)
    assert obs["remaining_money"].shape == (1,)
    assert obs["current_high_bid"].shape == (1,)
    assert obs["my_current_bid"].shape == (1,)
    assert obs["bids"].shape == (5,)
    assert obs["current_player_prestige"].shape == (5,)
    assert obs["potential_player_prestige"].shape == (5,)
    assert obs["available_money_cards"].shape == (10,)
    assert obs["cards_in_bid"].shape == (10,)


def test_observation_initial_values():
    """Test that initial observations have correct values."""
    env = DiscreteHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    obs = env.observe("player_0")

    assert obs["total_prestige"][0] == 0
    # remaining_special_cards depends on first card drawn (could be 3 or 4)
    assert obs["remaining_special_cards"][0] in [3, 4]
    assert obs["remaining_money"][0] == 55  # sum(1..10)
    assert obs["current_high_bid"][0] == 0
    assert obs["my_current_bid"][0] == 0
    assert np.all(obs["available_money_cards"] == 1.0), "Should have all cards"
    assert np.all(obs["cards_in_bid"] == 0.0), "No cards in bid initially"


def test_invalid_action_raises():
    """Test that invalid actions raise errors."""
    env = DiscreteHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    # Player 0 bids card 10
    env.step(10)

    # Player 1 tries to bid card 5 (doesn't beat 10)
    with pytest.raises(ValueError, match="must exceed"):
        env.step(5)


def test_auction_loops_correctly():
    """Test that auction correctly loops through players."""
    env = DiscreteHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    initial_round = env.game_state.cur_round.num

    # Player 0 bids 3
    assert env.agent_selection == "player_0"
    env.step(3)

    # Player 1 bids 4
    assert env.agent_selection == "player_1"
    env.step(4)

    # Player 2 bids 5
    assert env.agent_selection == "player_2"
    env.step(5)

    # Back to player 0
    assert env.agent_selection == "player_0"
    assert env.game_state.cur_round.num == initial_round, "Should still be same round"

    # Player 0 bids another card (4) to raise
    env.step(4)  # Total bid now 3 + 4 = 7

    # Player 1's turn
    assert env.agent_selection == "player_1"
