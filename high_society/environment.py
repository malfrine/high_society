from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector
from gymnasium import spaces
import numpy as np
from pydantic import BaseModel

from typing import Literal
import random

class PrestigeCard(BaseModel):
    type: Literal["value","special"]
    value: int | None = None
    speciality: Literal["2x"] | None = None

    def get_multiplier(self) -> float:
        if self.speciality == "2x":
            return 2
        else:
            return 1
    
class MoneyCard(BaseModel):
    value: float

class AuctionRound(BaseModel):
    num: int
    cur_bidder_idx: int
    cur_bid: float
    bids: dict[int, float]
    players_to_bid: set[int]
    card: PrestigeCard
    value_to_agent: dict[int, float]
    
class PlayerState(BaseModel):
    player_idx: int
    player_name: str
    money_cards: list[MoneyCard]
    prestige_cards: list[PrestigeCard]
    total_prestige: float = 0
    total_money: float = 0
    
class GameState(BaseModel):
    round_starter_idx: int = 0
    remaining_special_cards: int = 4
    player_states: dict[int, PlayerState]
    remaining_prestige_cards: list[PrestigeCard]
    cur_round: AuctionRound | None = None
        

def get_total_prestige(cards: list[PrestigeCard]) -> float:
    total_value = sum(card.value for card in cards if card.type == "value")
    multiplier = 1
    for card in cards:
        if card.type == "special":
            multiplier *= card.get_multiplier()
    return total_value * multiplier



class SimpleHighSocietyEnv(AECEnv):

    def __init__(self, num_players: int):
        super().__init__()
        if not (3 <= num_players <= 5):
            raise ValueError("Must have between 3 - 5 players")
        self.name = "simple_high_society"
        self.num_players = num_players
        self.agents = [f"player_{i}" for i in range(num_players)]
        self.possible_agents = self.agents[:]

        self.observation_spaces = {
            agent: spaces.Dict({
                "total_prestige": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "remaining_special_cards": spaces.Box(low=1, high=4, shape=(1,), dtype=np.float32),
                "is_last_round": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "remaining_money": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "current_round_bid": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "bids": spaces.Box(low=0, high=100, shape=(num_players,), dtype=np.float32),
                "current_player_prestige": spaces.Box(low=0, high=100, shape=(num_players,), dtype=np.float32),
                "potential_player_prestige": spaces.Box(low=0, high=100, shape=(num_players,), dtype=np.float32),
                "current_round_starter": spaces.Box(low=0, high=1, shape=(num_players,), dtype=np.float32),
                "next_round_starter": spaces.Box(low=0, high=1, shape=(num_players,), dtype=np.float32),
            })
            for agent in self.agents
        }
        # Action is raise_intensity in [0, 1]: 0 = pass, > 0 = raise
        self.action_spaces = {
            agent: spaces.Box(0, 1, shape=(1,), dtype=np.float32) for agent in self.agents
        }
        
        self.reset()

    def obs_dim(self, agent: str) -> int:
        return sum(space.shape[0] for space in self.observation_space(agent).values())

    def action_dim(self, agent: str) -> int:
        return self.action_space(agent).shape[0]
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)

        agent = self.agent_selection
        agent_idx = self.agents.index(agent)
        raise_intensity = float(action[0])

        # Convert raise_intensity to bid amount
        # raise_intensity = 0 -> pass
        # raise_intensity > 0 -> bid = min_bid + raise_intensity * (available_money - min_bid)
        if raise_intensity == 0:
            self._handle_pass(agent_idx)
        else:
            auction = self.game_state.cur_round
            player_state = self.game_state.player_states[agent_idx]
            available_money = player_state.total_money + auction.bids.get(agent_idx, 0)
            min_bid = auction.cur_bid + 1
            bid_amount = min_bid + raise_intensity * (available_money - min_bid)
            self._handle_bid(agent_idx, bid_amount)

        # Check if auction round is complete
        if len(self.game_state.cur_round.players_to_bid) <= 1:
            self._complete_auction_round()

            # Check if game is over
            if self._is_game_over():
                self._calculate_final_scores()
                return

            # Start next auction round
            self.game_state.cur_round = self.start_auction_round(self.game_state)

        # Select next agent
        self._select_next_agent()

        # Accumulate rewards (sparse - only at end)
        self._clear_rewards()

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.game_state = self.start_game(self.agents)

        # Start first auction round
        self.game_state.cur_round = self.start_auction_round(self.game_state)

        # Initialize AECEnv properties
        self.agents = self.possible_agents[:]
        self._agent_selector = self._create_agent_selector()

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Select first agent who can afford to bid
        self._select_first_valid_agent(self.game_state.round_starter_idx)

        return self.observe(self.agent_selection), self.infos[self.agent_selection]
        
    def render(self):
        raise NotImplementedError()

    def close(self):
        pass
    
    def start_game(self, players: list[str]) -> GameState:

        prestige_cards = [
            *[PrestigeCard(type="value", value=i) for i in range(1, 10)],
            *[PrestigeCard(type="special", value=None, speciality="2x") for _ in range(4)],
        ]
        random.shuffle(prestige_cards)

        player_states = {}
        for i, player_name in enumerate(players):
            money_cards = [MoneyCard(value=i) for i in range(1, 10)]
            player_states[i] = PlayerState(
                player_idx=i,
                player_name=player_name,
                money_cards=money_cards[:],
                prestige_cards=[],
                total_prestige=0,
                total_money=sum(money_card.value for money_card in money_cards)
            )

        return GameState(
            round_starter_idx=0,
            remaining_special_cards=4,
            player_states=player_states,
            remaining_prestige_cards=prestige_cards,
        )
        
    def start_auction_round(self, game_state: GameState) -> AuctionRound:
        drawn_card = game_state.remaining_prestige_cards.pop()

        # Update remaining special cards count
        if drawn_card.type == "special":
            game_state.remaining_special_cards -= 1

        value_to_agent = {}
        for player_idx, player_state in game_state.player_states.items():
            potential_prestige_cards = [drawn_card] + player_state.prestige_cards
            value_to_agent[player_idx] = get_total_prestige(potential_prestige_cards)

        round_num = game_state.cur_round.num + 1 if game_state.cur_round else 1

        return AuctionRound(
            num=round_num,
            cur_bidder_idx=game_state.round_starter_idx,
            cur_bid=0,
            bids={player_idx: 0 for player_idx in game_state.player_states.keys()},
            players_to_bid=set(game_state.player_states.keys()),
            card=drawn_card,
            value_to_agent=value_to_agent,
        )

    def _create_agent_selector(self):
        """Create agent selector that follows round starter order"""
        return AgentSelector(self.agents)

    def _handle_pass(self, agent_idx: int):
        """Handle a player passing on the current auction"""
        auction = self.game_state.cur_round
        player_state = self.game_state.player_states[agent_idx]

        # Return bid money to player
        if agent_idx in auction.bids:
            player_state.total_money += auction.bids[agent_idx]
            auction.bids[agent_idx] = 0

        # Remove from active bidders
        auction.players_to_bid.discard(agent_idx)

    def _handle_bid(self, agent_idx: int, bid_amount: float):
        """Handle a player making a bid"""
        auction = self.game_state.cur_round
        player_state = self.game_state.player_states[agent_idx]

        # Validate bid
        required_bid = auction.cur_bid + 1
        if bid_amount < required_bid:
            raise ValueError(f"Bid must be at least {required_bid}, got {bid_amount}")

        if bid_amount > player_state.total_money + auction.bids.get(agent_idx, 0):
            raise ValueError(f"Insufficient funds. Have {player_state.total_money}, bid {bid_amount}")

        # Return previous bid money
        if agent_idx in auction.bids:
            player_state.total_money += auction.bids[agent_idx]

        # Place new bid
        player_state.total_money -= bid_amount
        auction.bids[agent_idx] = bid_amount
        auction.cur_bid = bid_amount
        auction.cur_bidder_idx = agent_idx

    def _complete_auction_round(self):
        """Complete the auction round and award card to winner"""
        auction = self.game_state.cur_round

        # Check if anyone bid (if everyone passed, cur_bid will be 0)
        if auction.cur_bid > 0 and len(auction.players_to_bid) > 0:
            # Someone made a bid and is the last one standing
            winner_idx = auction.cur_bidder_idx
            winner_state = self.game_state.player_states[winner_idx]

            # Award card to winner (they already paid)
            winner_state.prestige_cards.append(auction.card)
            winner_state.total_prestige = get_total_prestige(winner_state.prestige_cards)

            # Return bids to all other players (should already be done, but just in case)
            for player_idx, bid in auction.bids.items():
                if player_idx != winner_idx and bid > 0:
                    self.game_state.player_states[player_idx].total_money += bid
        else:
            # Everyone passed - card goes to discard, return all bids
            for player_idx, bid in auction.bids.items():
                if bid > 0:
                    self.game_state.player_states[player_idx].total_money += bid

        # Advance round starter
        self.game_state.round_starter_idx = (self.game_state.round_starter_idx + 1) % self.num_players

    def _is_game_over(self) -> bool:
        """Check if game is over (4 special cards drawn)"""
        return self.game_state.remaining_special_cards == 0

    def _calculate_final_scores(self):
        """Calculate final scores with elimination rule"""
        # Find minimum money
        min_money = min(
            player.total_money for player in self.game_state.player_states.values()
        )

        # Eliminate players with minimum money
        eliminated_players = set()
        for player_idx, player_state in self.game_state.player_states.items():
            if player_state.total_money == min_money:
                eliminated_players.add(player_idx)

        # Calculate rewards (prestige for non-eliminated, 0 for eliminated)
        for player_idx, player_state in self.game_state.player_states.items():
            agent = self.agents[player_idx]
            if player_idx in eliminated_players:
                self.rewards[agent] = 0
            else:
                self.rewards[agent] = player_state.total_prestige

            self.terminations[agent] = True

    def _select_next_agent(self):
        """Select next agent to bid in auction."""
        current_idx = self.agents.index(self.agent_selection)
        next_idx = (current_idx + 1) % self.num_players
        self._select_first_valid_agent(next_idx)

    def _select_first_valid_agent(self, start_idx: int):
        """Select first agent starting from start_idx who can afford min_bid.

        Auto-passes players who can't afford. Handles auction completion
        and new round setup if needed.
        """
        auction = self.game_state.cur_round
        min_bid = auction.cur_bid + 1

        for i in range(self.num_players):
            current_idx = (start_idx + i) % self.num_players
            if current_idx in auction.players_to_bid:
                player_state = self.game_state.player_states[current_idx]
                # Include their current bid since they'd get it back to re-bid
                available_money = player_state.total_money + auction.bids.get(current_idx, 0)
                if available_money >= min_bid:
                    self.agent_selection = self.agents[current_idx]
                    return
                else:
                    # Auto-pass for players who can't afford min_bid
                    self._handle_pass(current_idx)

        # Check if auction ended due to auto-passes
        if len(auction.players_to_bid) <= 1:
            self._complete_auction_round()
            if self._is_game_over():
                self._calculate_final_scores()
                return
            self.game_state.cur_round = self.start_auction_round(self.game_state)
            # Recursively select for new round
            self._select_first_valid_agent(self.game_state.round_starter_idx)

    def _clear_rewards(self):
        """Clear rewards for all agents"""
        for agent in self.agents:
            self.rewards[agent] = 0

    def observe(self, agent: str):
        """Generate observation for a specific agent"""
        agent_idx = self.agents.index(agent)
        player_state = self.game_state.player_states[agent_idx]
        auction = self.game_state.cur_round

        # Available money includes current bid (returned if they re-bid or pass)
        available_money = player_state.total_money + auction.bids.get(agent_idx, 0)

        obs = {
            "total_prestige": np.array([player_state.total_prestige], dtype=np.float32),
            "remaining_special_cards": np.array([self.game_state.remaining_special_cards], dtype=np.float32),
            "is_last_round": np.array([1 if self.game_state.remaining_special_cards == 1 else 0], dtype=np.float32),
            "remaining_money": np.array([available_money], dtype=np.float32),
            "current_round_bid": np.array([auction.cur_bid], dtype=np.float32),
            "bids": np.array([auction.bids.get(i, 0) for i in range(self.num_players)], dtype=np.float32),
            "current_player_prestige": np.array(
                [self.game_state.player_states[i].total_prestige for i in range(self.num_players)],
                dtype=np.float32
            ),
            "potential_player_prestige": np.array(
                [auction.value_to_agent[i] for i in range(self.num_players)],
                dtype=np.float32
            ),
            "current_round_starter": np.array(
                [1 if i == self.game_state.round_starter_idx else 0 for i in range(self.num_players)],
                dtype=np.float32
            ),
            "next_round_starter": np.array(
                [1 if i == (self.game_state.round_starter_idx + 1) % self.num_players else 0 for i in range(self.num_players)],
                dtype=np.float32
            ),
        }

        return obs