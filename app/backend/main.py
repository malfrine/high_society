from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from high_society.agents import DQNAgent, DiscreteRandomPassAgent
from high_society.environments.discrete import (
    ACTION_PASS,
    DiscreteHighSocietyEnv,
)
from high_society.utils import cat_dict_array

from .schemas import (
    ActionLogEntry,
    ActionRequest,
    GameResponse,
    PlayerInfo,
)

# --- Cached DQN weights (loaded once at startup) ---
_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "experiments" / "results" / "pool" / "dqn_agent_v3.pth"
_cached_weights: dict | None = None
_cached_agents: dict[tuple[int, int], DQNAgent] = {}


def _get_weights() -> dict:
    global _cached_weights
    if _cached_weights is None:
        _cached_weights = torch.load(str(_WEIGHTS_PATH), weights_only=True, map_location="cpu")
    return _cached_weights


def _get_dqn_agent(player_id: int, num_players: int) -> DQNAgent:
    key = (player_id, num_players)
    if key not in _cached_agents:
        env_tmp = DiscreteHighSocietyEnv(num_players=num_players)
        agent = DQNAgent(
            player_id=player_id,
            num_actions=env_tmp.num_actions,
            obs_space=env_tmp.observation_space("player_0"),
            epsilon=0.0,
        )
        agent.q_net.load_state_dict(_get_weights())
        _cached_agents[key] = agent
    return _cached_agents[key]


def _get_random_agent(player_id: int) -> DiscreteRandomPassAgent:
    return DiscreteRandomPassAgent(player_id=player_id, pass_probability=0.4)


def _describe_action(player_name: str, action: int) -> str:
    if action == ACTION_PASS:
        return f"{player_name} passed"
    return f"{player_name} added card {action}"


def _build_players(env: DiscreteHighSocietyEnv, human_idx: int = 0) -> list[PlayerInfo]:
    auction = env.game_state.cur_round
    players = []
    for idx, ps in env.game_state.player_states.items():
        players.append(PlayerInfo(
            player_idx=idx,
            player_name="You" if idx == human_idx else ps.player_name,
            money_cards=[mc.value for mc in ps.money_cards],
            total_prestige=ps.total_prestige,
            total_money=ps.total_money,
            is_human=idx == human_idx,
            current_bid=auction.bids.get(idx, 0) if auction else 0,
            is_active_in_auction=idx in auction.players_to_bid if auction else False,
        ))
    return players


def _build_response(
    env: DiscreteHighSocietyEnv,
    current_agent_idx: int,
    action_log: list[ActionLogEntry],
    human_idx: int = 0,
) -> GameResponse:
    game_over = all(env.terminations.values())

    winner_idx = None
    eliminated_indices: list[int] = []
    if game_over:
        min_money = min(p.total_money for p in env.game_state.player_states.values())
        eliminated_indices = [
            idx for idx, p in env.game_state.player_states.items()
            if p.total_money == min_money
        ]
        max_prestige = -1.0
        for idx, p in env.game_state.player_states.items():
            if idx not in eliminated_indices and p.total_prestige > max_prestige:
                max_prestige = p.total_prestige
                winner_idx = idx

    action_mask: list[bool] = []
    if not game_over:
        agent_name = env.agents[current_agent_idx]
        mask = env.get_action_mask(agent_name)
        action_mask = mask.tolist()

    return GameResponse(
        game_state=env.game_state,
        current_agent_idx=current_agent_idx,
        action_mask=action_mask,
        action_log=action_log,
        game_over=game_over,
        winner_idx=winner_idx,
        eliminated_indices=eliminated_indices,
        players=_build_players(env, human_idx),
    )


def _run_robot_turns(
    env: DiscreteHighSocietyEnv,
    human_idx: int,
    robot_type: str,
    num_players: int,
) -> tuple[int, list[ActionLogEntry]]:
    """Run robot turns until it's the human's turn or game over.

    Returns (current_agent_idx, action_log).
    """
    log: list[ActionLogEntry] = []

    while not all(env.terminations.values()):
        agent_name = env.agent_selection
        agent_idx = env.agents.index(agent_name)

        if agent_idx == human_idx:
            break

        obs = cat_dict_array(env.observe(agent_name))
        mask = env.get_action_mask(agent_name)

        if robot_type == "dqn":
            robot = _get_dqn_agent(agent_idx, num_players)
        else:
            robot = _get_random_agent(agent_idx)

        action, _ = robot.get_action(obs, mask)
        desc = _describe_action(agent_name, action)
        log.append(ActionLogEntry(player_name=agent_name, action=action, description=desc))

        env.step(action)

    if all(env.terminations.values()):
        current_idx = 0  # game over, doesn't matter
    else:
        current_idx = env.agents.index(env.agent_selection)

    return current_idx, log


# --- FastAPI app ---
app = FastAPI(title="High Society RL")


@app.get("/api/new-game")
def new_game(
    num_players: int = Query(default=4, ge=3, le=5),
    robot_type: str = Query(default="dqn", pattern="^(dqn|random)$"),
) -> GameResponse:
    env = DiscreteHighSocietyEnv(num_players=num_players)
    env.reset(num_players=num_players)

    human_idx = 0

    # Run robot turns if human isn't first
    current_idx, log = _run_robot_turns(env, human_idx, robot_type, num_players)

    return _build_response(env, current_idx, log, human_idx)


@app.post("/api/action")
def submit_action(req: ActionRequest) -> GameResponse:
    num_players = len(req.game_state.player_states)
    if not 3 <= num_players <= 5:
        raise HTTPException(status_code=422, detail="num_players must be 3-5")

    try:
        env = DiscreteHighSocietyEnv(num_players=num_players)
        env.restore_from_state(req.game_state, req.current_agent_idx)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to restore game state: {e}")

    # Validate action
    agent_name = env.agent_selection
    mask = env.get_action_mask(agent_name)
    if req.action < 0 or req.action >= len(mask) or not mask[req.action]:
        raise HTTPException(status_code=422, detail=f"Invalid action {req.action}")

    env.step(req.action)

    human_idx = 0
    robot_type = req.robot_type

    current_idx, log = _run_robot_turns(env, human_idx, robot_type, num_players)

    return _build_response(env, current_idx, log, human_idx)


# Serve frontend static files in production
_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
