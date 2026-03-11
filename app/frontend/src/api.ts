import type { GameResponse } from "./types";

export async function fetchNewGame(
  numPlayers: number,
  robotType: string
): Promise<GameResponse> {
  const params = new URLSearchParams({
    num_players: String(numPlayers),
    robot_type: robotType,
  });
  const res = await fetch(`/api/new-game?${params}`);
  if (!res.ok) throw new Error(`New game failed: ${res.status}`);
  return res.json();
}

export async function submitAction(
  gameState: Record<string, unknown>,
  currentAgentIdx: number,
  action: number,
  robotType: string
): Promise<GameResponse> {
  const res = await fetch("/api/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      game_state: gameState,
      current_agent_idx: currentAgentIdx,
      action,
      robot_type: robotType,
    }),
  });
  if (!res.ok) throw new Error(`Action failed: ${res.status}`);
  return res.json();
}
