"""Tests for High Society auction mechanics"""
import pytest
import numpy as np
from high_society.environment import HighSocietyEnvSimple


def test_auction_round_loops_with_increasing_bids():
    """Test that the auction round continues (loops back) when bids increase.

    Scenario:
    - 2 players in an auction
    - Players alternate bidding higher amounts
    - Round should continue until someone passes
    """
    env = HighSocietyEnvSimple(name="test", num_players=2)
    env.reset(seed=42)

    # Get initial state
    initial_round_num = env.game_state.cur_round.num
    auction = env.game_state.cur_round

    # Track which agents have taken turns
    agents_that_acted = []

    # Player 0 bids 5
    assert env.agent_selection == "player_0"
    agents_that_acted.append(env.agent_selection)
    env.step(np.array([5.0]))

    # Round should still be active
    assert env.game_state.cur_round.num == initial_round_num
    assert env.game_state.cur_round.cur_bid == 5.0
    assert len(env.game_state.cur_round.players_to_bid) == 2  # Both still in

    # Player 1 bids 10
    assert env.agent_selection == "player_1"
    agents_that_acted.append(env.agent_selection)
    env.step(np.array([10.0]))

    # Round should still be active
    assert env.game_state.cur_round.num == initial_round_num
    assert env.game_state.cur_round.cur_bid == 10.0
    assert len(env.game_state.cur_round.players_to_bid) == 2  # Both still in

    # Player 0 bids 15 (round loops back to player_0)
    assert env.agent_selection == "player_0"
    agents_that_acted.append(env.agent_selection)
    env.step(np.array([15.0]))

    # Round should still be active and looping
    assert env.game_state.cur_round.num == initial_round_num
    assert env.game_state.cur_round.cur_bid == 15.0
    assert len(env.game_state.cur_round.players_to_bid) == 2

    # Verify that player_0 has acted multiple times (proving the loop)
    assert agents_that_acted.count("player_0") == 2
    assert agents_that_acted.count("player_1") == 1

    # Now player 1 passes
    assert env.agent_selection == "player_1"
    env.step(np.array([0.0]))  # Pass

    # Round should complete and move to next round
    assert env.game_state.cur_round.num == initial_round_num + 1
    # Player 0 should have won the card
    assert len(env.game_state.player_states[0].prestige_cards) == 1
    assert len(env.game_state.player_states[1].prestige_cards) == 0
    # Player 0 should have paid 15
    assert env.game_state.player_states[0].total_money == 45 - 15


def test_auction_round_terminates_when_all_pass():
    """Test that auction round terminates and card is discarded when no one bids.

    Scenario:
    - 2 players
    - Player 0 passes without bidding
    - Only 1 player remains, but since cur_bid is 0, card is discarded
    """
    env = HighSocietyEnvSimple(name="test", num_players=2)
    env.reset(seed=42)

    initial_round_num = env.game_state.cur_round.num

    # Player 0 passes
    assert env.agent_selection == "player_0"
    env.step(np.array([0.0]))

    # Round should complete immediately when only 1 player remains
    assert env.game_state.cur_round.num == initial_round_num + 1

    # No one bid, so card is discarded - no one wins
    assert len(env.game_state.player_states[0].prestige_cards) == 0
    assert len(env.game_state.player_states[1].prestige_cards) == 0

    # Both players should still have full money (no actual bids)
    assert env.game_state.player_states[0].total_money == 45
    assert env.game_state.player_states[1].total_money == 45


def test_auction_round_terminates_with_single_winner():
    """Test that auction round terminates correctly with a single winner.

    Scenario:
    - 2 players
    - Player 0 bids, player 1 passes
    - Player 0 should win immediately
    """
    env = HighSocietyEnvSimple(name="test", num_players=2)
    env.reset(seed=42)

    initial_round_num = env.game_state.cur_round.num

    # Player 0 bids 20
    assert env.agent_selection == "player_0"
    env.step(np.array([20.0]))

    # Round still active, waiting for player 1
    assert env.game_state.cur_round.num == initial_round_num
    assert len(env.game_state.cur_round.players_to_bid) == 2

    # Player 1 passes
    assert env.agent_selection == "player_1"
    env.step(np.array([0.0]))

    # Round should complete immediately
    assert env.game_state.cur_round.num == initial_round_num + 1

    # Player 0 should have won the card
    assert len(env.game_state.player_states[0].prestige_cards) == 1
    assert len(env.game_state.player_states[1].prestige_cards) == 0

    # Player 0 should have paid their bid
    assert env.game_state.player_states[0].total_money == 45 - 20
    # Player 1 should have full money
    assert env.game_state.player_states[1].total_money == 45


def test_auction_with_three_players_multiple_rounds():
    """Test auction mechanics with 3 players through multiple bidding rounds.

    Scenario:
    - 3 players
    - Player 0 bids 5
    - Player 1 bids 10
    - Player 2 passes
    - Player 0 bids 15
    - Player 1 passes
    - Player 0 wins
    """
    env = HighSocietyEnvSimple(name="test", num_players=3)
    env.reset(seed=123)

    initial_round_num = env.game_state.cur_round.num

    # Player 0 bids 5
    assert env.agent_selection == "player_0"
    env.step(np.array([5.0]))
    assert len(env.game_state.cur_round.players_to_bid) == 3

    # Player 1 bids 10
    assert env.agent_selection == "player_1"
    env.step(np.array([10.0]))
    assert len(env.game_state.cur_round.players_to_bid) == 3

    # Player 2 passes
    assert env.agent_selection == "player_2"
    env.step(np.array([0.0]))
    assert len(env.game_state.cur_round.players_to_bid) == 2  # Only 0 and 1 left

    # Should loop back to player 0
    assert env.agent_selection == "player_0"
    env.step(np.array([15.0]))
    assert len(env.game_state.cur_round.players_to_bid) == 2

    # Player 1 passes
    assert env.agent_selection == "player_1"
    env.step(np.array([0.0]))

    # Round should complete
    assert env.game_state.cur_round.num == initial_round_num + 1

    # Player 0 wins
    assert len(env.game_state.player_states[0].prestige_cards) == 1
    assert len(env.game_state.player_states[1].prestige_cards) == 0
    assert len(env.game_state.player_states[2].prestige_cards) == 0

    # Check money
    assert env.game_state.player_states[0].total_money == 45 - 15
    assert env.game_state.player_states[1].total_money == 45  # Bid returned
    assert env.game_state.player_states[2].total_money == 45  # Never bid
