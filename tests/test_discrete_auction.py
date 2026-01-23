"""Tests for DiscreteHighSocietyEnv auction mechanics"""
import pytest
import numpy as np
from high_society.environments.discrete import DiscreteHighSocietyEnv, ACTION_PASS


def test_auction_round_loops_with_increasing_bids():
    """Test that the auction round continues (loops back) when bids increase."""
    env = DiscreteHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    initial_round_num = env.game_state.cur_round.num
    agents_that_acted = []

    # Player 0 bids card 3
    assert env.agent_selection == "player_0"
    agents_that_acted.append(env.agent_selection)
    env.step(3)

    assert env.game_state.cur_round.num == initial_round_num
    assert env.game_state.cur_round.cur_bid == 3
    assert len(env.game_state.cur_round.players_to_bid) == 3

    # Player 1 bids card 4
    assert env.agent_selection == "player_1"
    agents_that_acted.append(env.agent_selection)
    env.step(4)

    assert env.game_state.cur_round.cur_bid == 4
    assert len(env.game_state.cur_round.players_to_bid) == 3

    # Player 2 bids card 5
    assert env.agent_selection == "player_2"
    agents_that_acted.append(env.agent_selection)
    env.step(5)

    assert env.game_state.cur_round.cur_bid == 5
    assert len(env.game_state.cur_round.players_to_bid) == 3

    # Player 0 raises again (adds card 6, total bid = 3 + 6 = 9)
    assert env.agent_selection == "player_0"
    agents_that_acted.append(env.agent_selection)
    env.step(6)

    assert env.game_state.cur_round.num == initial_round_num
    assert env.game_state.cur_round.cur_bid == 9  # 3 + 6
    assert len(env.game_state.cur_round.players_to_bid) == 3

    # Verify player_0 acted multiple times (proving the loop)
    assert agents_that_acted.count("player_0") == 2

    # Player 1 passes
    assert env.agent_selection == "player_1"
    env.step(ACTION_PASS)
    assert len(env.game_state.cur_round.players_to_bid) == 2

    # Player 2 passes
    assert env.agent_selection == "player_2"
    env.step(ACTION_PASS)

    # Round should complete
    assert env.game_state.cur_round.num == initial_round_num + 1
    assert len(env.game_state.player_states[0].prestige_cards) == 1
    assert len(env.game_state.player_states[1].prestige_cards) == 0
    assert len(env.game_state.player_states[2].prestige_cards) == 0


def test_auction_round_terminates_when_all_pass():
    """Test that auction terminates and card is discarded when no one bids."""
    env = DiscreteHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    initial_round_num = env.game_state.cur_round.num

    # Player 0 passes
    assert env.agent_selection == "player_0"
    env.step(ACTION_PASS)
    assert len(env.game_state.cur_round.players_to_bid) == 2

    # Player 1 passes
    assert env.agent_selection == "player_1"
    env.step(ACTION_PASS)

    # Round completes when only 1 player remains
    assert env.game_state.cur_round.num == initial_round_num + 1

    # No one bid, card discarded
    assert len(env.game_state.player_states[0].prestige_cards) == 0
    assert len(env.game_state.player_states[1].prestige_cards) == 0
    assert len(env.game_state.player_states[2].prestige_cards) == 0

    # All players should still have full money
    assert env.game_state.player_states[0].total_money == 55
    assert env.game_state.player_states[1].total_money == 55
    assert env.game_state.player_states[2].total_money == 55


def test_auction_round_terminates_with_single_winner():
    """Test that auction terminates correctly with a single winner."""
    env = DiscreteHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    initial_round_num = env.game_state.cur_round.num

    # Player 0 bids card 5
    assert env.agent_selection == "player_0"
    env.step(5)
    assert env.game_state.cur_round.cur_bid == 5

    # Player 1 passes
    assert env.agent_selection == "player_1"
    env.step(ACTION_PASS)
    assert len(env.game_state.cur_round.players_to_bid) == 2

    # Player 2 passes
    assert env.agent_selection == "player_2"
    env.step(ACTION_PASS)

    # Round completes
    assert env.game_state.cur_round.num == initial_round_num + 1

    # Player 0 won
    assert len(env.game_state.player_states[0].prestige_cards) == 1
    assert len(env.game_state.player_states[1].prestige_cards) == 0
    assert len(env.game_state.player_states[2].prestige_cards) == 0

    # Player 0 paid card 5
    assert env.game_state.player_states[0].total_money == 55 - 5
    assert env.game_state.player_states[1].total_money == 55
    assert env.game_state.player_states[2].total_money == 55


def test_auction_with_multiple_cards_in_bid():
    """Test auction where players add multiple cards to their bids."""
    env = DiscreteHighSocietyEnv(num_players=3)
    env.reset(seed=123)

    initial_round_num = env.game_state.cur_round.num

    # Player 0 bids card 2
    assert env.agent_selection == "player_0"
    env.step(2)
    assert env.game_state.cur_round.bids[0] == 2

    # Player 1 bids card 3
    assert env.agent_selection == "player_1"
    env.step(3)
    assert env.game_state.cur_round.bids[1] == 3

    # Player 2 passes
    assert env.agent_selection == "player_2"
    env.step(ACTION_PASS)
    assert len(env.game_state.cur_round.players_to_bid) == 2

    # Player 0 adds card 4 (total = 2 + 4 = 6)
    assert env.agent_selection == "player_0"
    env.step(4)
    assert env.game_state.cur_round.bids[0] == 6
    assert env.game_state.cur_round.cards_in_bid[0] == {2, 4}

    # Player 1 adds card 7 (total = 3 + 7 = 10)
    assert env.agent_selection == "player_1"
    env.step(7)
    assert env.game_state.cur_round.bids[1] == 10
    assert env.game_state.cur_round.cards_in_bid[1] == {3, 7}

    # Player 0 passes
    assert env.agent_selection == "player_0"
    env.step(ACTION_PASS)

    # Round completes, player 1 wins
    assert env.game_state.cur_round.num == initial_round_num + 1
    assert len(env.game_state.player_states[1].prestige_cards) == 1

    # Player 1 lost cards 3 and 7
    assert env.game_state.player_states[1].total_money == 55 - 3 - 7

    # Player 0 got cards 2 and 4 back
    assert env.game_state.player_states[0].total_money == 55
    player_0_card_values = {mc.value for mc in env.game_state.player_states[0].money_cards}
    assert 2 in player_0_card_values
    assert 4 in player_0_card_values
