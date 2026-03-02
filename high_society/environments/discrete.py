from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector
from gymnasium import spaces
import numpy as np
from pydantic import BaseModel

from typing import Literal
import random


class PrestigeCard(BaseModel):
    type: Literal["value", "special"]
    value: int | None = None
    speciality: Literal["2x"] | None = None

    def get_multiplier(self) -> float:
        if self.speciality == "2x":
            return 2
        else:
            return 1


class MoneyCard(BaseModel):
    value: int


class AuctionRound(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    num: int
    cur_bidder_idx: int
    cur_bid: int
    bids: dict[int, int]  # player_idx -> total bid value
    cards_in_bid: dict[int, set[int]]  # player_idx -> set of card values in bid
    players_to_bid: set[int]
    card: PrestigeCard
    value_to_agent: dict[int, float]


class PlayerState(BaseModel):
    player_idx: int
    player_name: str
    money_cards: list[MoneyCard]
    prestige_cards: list[PrestigeCard]
    total_prestige: float = 0
    total_money: int = 0


class GameState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

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


# Money card values
MONEY_CARD_VALUES = tuple(range(1, 11))  # 1-10
NUM_MONEY_CARDS = len(MONEY_CARD_VALUES)

# Action indices: 0 = PASS, 1-10 = add money card of that value
ACTION_PASS = 0

MAX_NUM_PLAYERS = 5 

class DiscreteHighSocietyEnv(AECEnv):
    """High Society environment with discrete action space.

    Simplified rules: each turn you add exactly ONE card to your bid.

    Actions:
        0: Pass (exit the auction, get your bid cards back)
        1-10: Add money card of value 1-10 to your bid

    A card can only be added if it makes your bid strictly greater than
    the current high bid.

    Players have money cards [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].
    Cards are only permanently spent when you WIN an auction.
    """

    def __init__(self, num_players: int = None):
        super().__init__()
        self.name = "discrete_high_society"
        self.reset(num_players or MAX_NUM_PLAYERS)

    def obs_dim(self, agent: str) -> int:
        return sum(space.shape[0] for space in self.observation_space(agent).values())

    def action_dim(self, agent: str) -> int:
        return self.num_actions

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def get_action_mask(self, agent: str) -> np.ndarray:
        """Get mask of valid actions for the agent.

        Returns:
            Boolean array of shape (num_actions,) where True = valid action
        """
        agent_idx = self.agents.index(agent)
        player_state = self.game_state.player_states[agent_idx]
        auction = self.game_state.cur_round

        mask = np.zeros(self.num_actions, dtype=bool)

        # Can always pass
        mask[ACTION_PASS] = True

        my_bid = auction.bids.get(agent_idx, 0)
        need_to_beat = auction.cur_bid

        for card_value in MONEY_CARD_VALUES:
            action_idx = card_value  # action 1 = card value 1, etc.

            # Can use this card if:
            # 1. Player still has the card (not spent in previous auctions)
            has_card = any(mc.value == card_value for mc in player_state.money_cards)
            # 2. Card is not already in current bid
            in_current_bid = card_value in auction.cards_in_bid.get(agent_idx, set())
            # 3. Adding this card makes bid strictly greater than current high bid
            new_bid_would_be = my_bid + card_value
            would_beat = new_bid_would_be > need_to_beat

            if has_card and not in_current_bid and would_beat:
                mask[action_idx] = True

        return mask

    def step(self, action: int):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)

        agent = self.agent_selection
        agent_idx = self.agents.index(agent)

        if action == ACTION_PASS:
            self._handle_pass(agent_idx)
        else:
            card_value = action  # action 1 = card value 1, etc.
            self._handle_add_card(agent_idx, card_value)

        # Check if auction round is complete (only 1 player left)
        if len(self.game_state.cur_round.players_to_bid) <= 1:
            self._complete_auction_round()

            if self._is_game_over():
                self._calculate_final_scores()
                return

            self.game_state.cur_round = self._start_auction_round()

        self._select_next_agent()
        self._clear_rewards()

    def reset(self, num_players = None, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        if num_players is not None:
            self.num_players = num_players
        
        assert self.num_players is not None
        assert 3 <= self.num_players <= 5
        self.agents = [f"player_{i}" for i in range(self.num_players)]
        self.possible_agents = self.agents[:]
        self.num_actions = 1 + NUM_MONEY_CARDS  # PASS + 10 money cards

        self.observation_spaces = {
            agent: spaces.Dict({
                "total_prestige": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "remaining_special_cards": spaces.Box(low=0, high=4, shape=(1,), dtype=np.float32),
                "is_last_round": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "remaining_money": spaces.Box(low=0, high=55, shape=(1,), dtype=np.float32),
                "current_high_bid": spaces.Box(low=0, high=55, shape=(1,), dtype=np.float32),
                "my_current_bid": spaces.Box(low=0, high=55, shape=(1,), dtype=np.float32),
                "bids": spaces.Box(low=0, high=55, shape=(MAX_NUM_PLAYERS,), dtype=np.float32),
                "current_player_prestige": spaces.Box(low=0, high=100, shape=(MAX_NUM_PLAYERS,), dtype=np.float32),
                "potential_player_prestige": spaces.Box(low=0, high=100, shape=(MAX_NUM_PLAYERS,), dtype=np.float32),
                # Which money cards the player still has (1 = has, 0 = spent in previous auctions)
                "available_money_cards": spaces.Box(low=0, high=1, shape=(NUM_MONEY_CARDS,), dtype=np.float32),
                # Which cards are committed in current bid
                "cards_in_bid": spaces.Box(low=0, high=1, shape=(NUM_MONEY_CARDS,), dtype=np.float32),
            })
            for agent in self.agents
        }

        self.action_spaces = {
            agent: spaces.Discrete(self.num_actions) for agent in self.agents
        }

        self.game_state = self._start_game()
        self.game_state.cur_round = self._start_auction_round()

        self.agents = self.possible_agents[:]
        self._agent_selector = AgentSelector(self.agents)

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._select_first_valid_agent(self.game_state.round_starter_idx)

        return self.observe(self.agent_selection), self.infos[self.agent_selection]

    def render(self):
        raise NotImplementedError()

    def close(self):
        pass

    def _start_game(self) -> GameState:
        prestige_cards = [
            *[PrestigeCard(type="value", value=i) for i in range(1, 10)],
            *[PrestigeCard(type="special", value=None, speciality="2x") for _ in range(4)],
        ]
        random.shuffle(prestige_cards)

        player_states = {}
        for i in range(self.num_players):
            money_cards = [MoneyCard(value=v) for v in MONEY_CARD_VALUES]
            player_states[i] = PlayerState(
                player_idx=i,
                player_name=f"player_{i}",
                money_cards=money_cards,
                prestige_cards=[],
                total_prestige=0,
                total_money=sum(MONEY_CARD_VALUES)  # 55
            )

        return GameState(
            round_starter_idx=0,
            remaining_special_cards=4,
            player_states=player_states,
            remaining_prestige_cards=prestige_cards,
        )

    def _start_auction_round(self) -> AuctionRound:
        drawn_card = self.game_state.remaining_prestige_cards.pop()

        if drawn_card.type == "special":
            self.game_state.remaining_special_cards -= 1

        value_to_agent = {}
        for player_idx, player_state in self.game_state.player_states.items():
            potential_cards = [drawn_card] + player_state.prestige_cards
            value_to_agent[player_idx] = get_total_prestige(potential_cards)

        round_num = self.game_state.cur_round.num + 1 if self.game_state.cur_round else 1

        return AuctionRound(
            num=round_num,
            cur_bidder_idx=self.game_state.round_starter_idx,
            cur_bid=0,
            bids={i: 0 for i in range(self.num_players)},
            cards_in_bid={i: set() for i in range(self.num_players)},
            players_to_bid=set(range(self.num_players)),
            card=drawn_card,
            value_to_agent=value_to_agent,
        )

    def _handle_pass(self, agent_idx: int):
        """Player passes - they exit auction and get bid cards back."""
        auction = self.game_state.cur_round

        # Cards in bid go back to being available (they're still in money_cards)
        # Just clear the bid tracking
        auction.bids[agent_idx] = 0
        auction.cards_in_bid[agent_idx] = set()

        auction.players_to_bid.discard(agent_idx)

    def _handle_add_card(self, agent_idx: int, card_value: int):
        """Player adds a card to their bid."""
        auction = self.game_state.cur_round
        player_state = self.game_state.player_states[agent_idx]

        # Validate
        has_card = any(mc.value == card_value for mc in player_state.money_cards)
        if not has_card:
            raise ValueError(f"Player {agent_idx} doesn't have card {card_value}")

        in_bid = card_value in auction.cards_in_bid[agent_idx]
        if in_bid:
            raise ValueError(f"Card {card_value} already in bid")

        new_bid = auction.bids[agent_idx] + card_value
        if new_bid <= auction.cur_bid:
            raise ValueError(f"New bid {new_bid} must exceed {auction.cur_bid}")

        # Add card to bid
        auction.cards_in_bid[agent_idx].add(card_value)
        auction.bids[agent_idx] = new_bid
        auction.cur_bid = new_bid
        auction.cur_bidder_idx = agent_idx

    def _complete_auction_round(self):
        """Complete auction and award card to winner."""
        auction = self.game_state.cur_round

        if auction.cur_bid > 0 and len(auction.players_to_bid) > 0:
            winner_idx = auction.cur_bidder_idx
            winner_state = self.game_state.player_states[winner_idx]

            # Award prestige card
            winner_state.prestige_cards.append(auction.card)
            winner_state.total_prestige = get_total_prestige(winner_state.prestige_cards)

            # Remove spent money cards permanently
            spent_values = auction.cards_in_bid[winner_idx]
            winner_state.money_cards = [
                mc for mc in winner_state.money_cards
                if mc.value not in spent_values
            ]
            winner_state.total_money = sum(mc.value for mc in winner_state.money_cards)

        self.game_state.round_starter_idx = (self.game_state.round_starter_idx + 1) % self.num_players

    def _is_game_over(self) -> bool:
        return self.game_state.remaining_special_cards == 0

    def _calculate_final_scores(self):
        """Elimination rule: lowest money is eliminated, highest prestige wins."""
        min_money = min(p.total_money for p in self.game_state.player_states.values())

        eliminated = {
            idx for idx, p in self.game_state.player_states.items()
            if p.total_money == min_money
        }

        # Find winner among non-eliminated
        winner_idx = None
        max_prestige = -1
        for idx, p in self.game_state.player_states.items():
            if idx not in eliminated and p.total_prestige > max_prestige:
                max_prestige = p.total_prestige
                winner_idx = idx

        for idx in range(self.num_players):
            agent = self.agents[idx]
            self.rewards[agent] = 1.0 if idx == winner_idx else -1.0
            self.terminations[agent] = True

    def _select_next_agent(self):
        current_idx = self.agents.index(self.agent_selection)
        next_idx = (current_idx + 1) % self.num_players
        self._select_first_valid_agent(next_idx)

    def _select_first_valid_agent(self, start_idx: int):
        """Select next agent who is still in the auction."""
        auction = self.game_state.cur_round

        for i in range(self.num_players):
            idx = (start_idx + i) % self.num_players
            if idx in auction.players_to_bid:
                self.agent_selection = self.agents[idx]
                return

        # No one left - auction should have ended
        if len(auction.players_to_bid) <= 1:
            self._complete_auction_round()
            if self._is_game_over():
                self._calculate_final_scores()
                return
            self.game_state.cur_round = self._start_auction_round()
            self._select_first_valid_agent(self.game_state.round_starter_idx)

    def _clear_rewards(self):
        for agent in self.agents:
            self.rewards[agent] = 0

    def observe(self, agent: str):
        agent_idx = self.agents.index(agent)
        player_state = self.game_state.player_states[agent_idx]
        auction = self.game_state.cur_round

        # Available money cards (not spent in previous auctions)
        available = np.zeros(NUM_MONEY_CARDS, dtype=np.float32)
        for mc in player_state.money_cards:
            available[mc.value - 1] = 1.0  # card value 1 -> index 0

        # Cards currently in bid
        in_bid = np.zeros(NUM_MONEY_CARDS, dtype=np.float32)
        for cv in auction.cards_in_bid.get(agent_idx, set()):
            in_bid[cv - 1] = 1.0

        # Pad arrays to MAX_NUM_PLAYERS so observation shape is consistent
        # regardless of actual number of players in the game
        bids = np.zeros(MAX_NUM_PLAYERS, dtype=np.float32)
        current_player_prestige = np.zeros(MAX_NUM_PLAYERS, dtype=np.float32)
        potential_player_prestige = np.zeros(MAX_NUM_PLAYERS, dtype=np.float32)

        for i in range(self.num_players):
            bids[i] = auction.bids.get(i, 0)
            current_player_prestige[i] = self.game_state.player_states[i].total_prestige
            potential_player_prestige[i] = auction.value_to_agent[i]

        return {
            "total_prestige": np.array([player_state.total_prestige], dtype=np.float32),
            "remaining_special_cards": np.array([self.game_state.remaining_special_cards], dtype=np.float32),
            "is_last_round": np.array([1.0 if self.game_state.remaining_special_cards == 1 else 0.0], dtype=np.float32),
            "remaining_money": np.array([player_state.total_money], dtype=np.float32),
            "current_high_bid": np.array([auction.cur_bid], dtype=np.float32),
            "my_current_bid": np.array([auction.bids.get(agent_idx, 0)], dtype=np.float32),
            "bids": bids,
            "current_player_prestige": current_player_prestige,
            "potential_player_prestige": potential_player_prestige,
            "available_money_cards": available,
            "cards_in_bid": in_bid,
        }
