"""Tests for High Society auction mechanics"""
import pytest
import numpy as np
from high_society.environment import SimpleHighSocietyEnv


def test_auction_round_loops_with_increasing_bids():
    """Test that the auction round continues (loops back) when bids increase.

    Scenario:
    - 3 players in an auction
    - Players alternate raising
    - Round should continue until only one player remains
    """
    env = SimpleHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    # Get initial state
    initial_round_num = env.game_state.cur_round.num

    # Track which agents have taken turns
    agents_that_acted = []

    # Player 0 raises (small raise)
    assert env.agent_selection == "player_0"
    agents_that_acted.append(env.agent_selection)
    env.step(np.array([0.1]))  # 10% raise intensity

    # Round should still be active
    assert env.game_state.cur_round.num == initial_round_num
    assert env.game_state.cur_round.cur_bid > 0
    assert len(env.game_state.cur_round.players_to_bid) == 3  # All still in
    bid_after_p0 = env.game_state.cur_round.cur_bid

    # Player 1 raises higher
    assert env.agent_selection == "player_1"
    agents_that_acted.append(env.agent_selection)
    env.step(np.array([0.2]))  # 20% raise intensity

    # Round should still be active
    assert env.game_state.cur_round.num == initial_round_num
    assert env.game_state.cur_round.cur_bid > bid_after_p0
    assert len(env.game_state.cur_round.players_to_bid) == 3  # All still in
    bid_after_p1 = env.game_state.cur_round.cur_bid

    # Player 2 raises higher
    assert env.agent_selection == "player_2"
    agents_that_acted.append(env.agent_selection)
    env.step(np.array([0.3]))

    # Round should still be active
    assert env.game_state.cur_round.num == initial_round_num
    assert env.game_state.cur_round.cur_bid > bid_after_p1
    assert len(env.game_state.cur_round.players_to_bid) == 3
    bid_after_p2 = env.game_state.cur_round.cur_bid

    # Player 0 raises again (round loops back to player_0)
    assert env.agent_selection == "player_0"
    agents_that_acted.append(env.agent_selection)
    env.step(np.array([0.4]))

    # Round should still be active and looping
    assert env.game_state.cur_round.num == initial_round_num
    assert env.game_state.cur_round.cur_bid > bid_after_p2
    assert len(env.game_state.cur_round.players_to_bid) == 3

    # Verify that player_0 has acted multiple times (proving the loop)
    assert agents_that_acted.count("player_0") == 2
    assert agents_that_acted.count("player_1") == 1
    assert agents_that_acted.count("player_2") == 1

    # Player 1 passes
    assert env.agent_selection == "player_1"
    env.step(np.array([0.0]))
    assert len(env.game_state.cur_round.players_to_bid) == 2  # Now 2 left

    # Player 2 passes
    assert env.agent_selection == "player_2"
    env.step(np.array([0.0]))

    # Round should complete and move to next round
    assert env.game_state.cur_round.num == initial_round_num + 1
    # Player 0 should have won the card
    assert len(env.game_state.player_states[0].prestige_cards) == 1
    assert len(env.game_state.player_states[1].prestige_cards) == 0
    assert len(env.game_state.player_states[2].prestige_cards) == 0
    # Player 0 should have paid (money decreased)
    assert env.game_state.player_states[0].total_money < 45


def test_auction_round_terminates_when_all_pass():
    """Test that auction round terminates and card is discarded when no one bids.

    Scenario:
    - 3 players
    - Player 0 passes, Player 1 passes
    - Only 1 player remains, but since cur_bid is 0, card is discarded
    """
    env = SimpleHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    initial_round_num = env.game_state.cur_round.num

    # Player 0 passes
    assert env.agent_selection == "player_0"
    env.step(np.array([0.0]))
    assert len(env.game_state.cur_round.players_to_bid) == 2  # 2 left

    # Player 1 passes
    assert env.agent_selection == "player_1"
    env.step(np.array([0.0]))

    # Round should complete when only 1 player remains
    assert env.game_state.cur_round.num == initial_round_num + 1

    # No one bid, so card is discarded - no one wins
    assert len(env.game_state.player_states[0].prestige_cards) == 0
    assert len(env.game_state.player_states[1].prestige_cards) == 0
    assert len(env.game_state.player_states[2].prestige_cards) == 0

    # All players should still have full money (no actual bids)
    assert env.game_state.player_states[0].total_money == 45
    assert env.game_state.player_states[1].total_money == 45
    assert env.game_state.player_states[2].total_money == 45


def test_auction_round_terminates_with_single_winner():
    """Test that auction round terminates correctly with a single winner.

    Scenario:
    - 3 players
    - Player 0 bids, player 1 passes, player 2 passes
    - Player 0 should win
    """
    env = SimpleHighSocietyEnv(num_players=3)
    env.reset(seed=42)

    initial_round_num = env.game_state.cur_round.num

    # Player 0 raises
    assert env.agent_selection == "player_0"
    env.step(np.array([0.5]))  # 50% raise intensity
    winning_bid = env.game_state.cur_round.cur_bid

    # Round still active, waiting for others
    assert env.game_state.cur_round.num == initial_round_num
    assert len(env.game_state.cur_round.players_to_bid) == 3

    # Player 1 passes
    assert env.agent_selection == "player_1"
    env.step(np.array([0.0]))
    assert len(env.game_state.cur_round.players_to_bid) == 2

    # Player 2 passes
    assert env.agent_selection == "player_2"
    env.step(np.array([0.0]))

    # Round should complete
    assert env.game_state.cur_round.num == initial_round_num + 1

    # Player 0 should have won the card
    assert len(env.game_state.player_states[0].prestige_cards) == 1
    assert len(env.game_state.player_states[1].prestige_cards) == 0
    assert len(env.game_state.player_states[2].prestige_cards) == 0

    # Player 0 should have paid their bid
    assert env.game_state.player_states[0].total_money == 45 - winning_bid
    # Others should have full money
    assert env.game_state.player_states[1].total_money == 45
    assert env.game_state.player_states[2].total_money == 45


def test_auction_with_three_players_multiple_rounds():
    """Test auction mechanics with 3 players through multiple bidding rounds.

    Scenario:
    - 3 players
    - Player 0 raises
    - Player 1 raises higher
    - Player 2 passes
    - Player 0 raises again
    - Player 1 passes
    - Player 0 wins
    """
    env = SimpleHighSocietyEnv(num_players=3)
    env.reset(seed=123)

    initial_round_num = env.game_state.cur_round.num

    # Player 0 raises
    assert env.agent_selection == "player_0"
    env.step(np.array([0.1]))
    bid_p0_first = env.game_state.cur_round.cur_bid
    assert len(env.game_state.cur_round.players_to_bid) == 3

    # Player 1 raises higher
    assert env.agent_selection == "player_1"
    env.step(np.array([0.2]))
    bid_p1 = env.game_state.cur_round.cur_bid
    assert bid_p1 > bid_p0_first
    assert len(env.game_state.cur_round.players_to_bid) == 3

    # Player 2 passes
    assert env.agent_selection == "player_2"
    env.step(np.array([0.0]))
    assert len(env.game_state.cur_round.players_to_bid) == 2  # Only 0 and 1 left

    # Should loop back to player 0
    assert env.agent_selection == "player_0"
    env.step(np.array([0.3]))
    final_bid = env.game_state.cur_round.cur_bid
    assert final_bid > bid_p1
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

    # Check money - player 0 paid the final bid
    assert env.game_state.player_states[0].total_money == 45 - final_bid
    assert env.game_state.player_states[1].total_money == 45  # Bid returned
    assert env.game_state.player_states[2].total_money == 45  # Never bid
