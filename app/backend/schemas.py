from pydantic import BaseModel

from high_society.environments.discrete import GameState


class NewGameParams(BaseModel):
    num_players: int = 4
    robot_type: str = "dqn"


class ActionRequest(BaseModel):
    game_state: GameState
    current_agent_idx: int
    action: int
    robot_type: str = "dqn"


class PlayerInfo(BaseModel):
    player_idx: int
    player_name: str
    money_cards: list[int]
    total_prestige: float
    total_money: int
    is_human: bool
    current_bid: int
    is_active_in_auction: bool


class ActionLogEntry(BaseModel):
    player_name: str
    action: int
    description: str


class GameResponse(BaseModel):
    game_state: GameState
    current_agent_idx: int
    action_mask: list[bool]
    action_log: list[ActionLogEntry]
    game_over: bool
    winner_idx: int | None
    eliminated_indices: list[int]
    players: list[PlayerInfo]
