export interface PlayerInfo {
  player_idx: number;
  player_name: string;
  money_cards: number[];
  total_prestige: number;
  total_money: number;
  is_human: boolean;
  current_bid: number;
  is_active_in_auction: boolean;
}

export interface ActionLogEntry {
  player_name: string;
  action: number;
  description: string;
}

export interface GameResponse {
  game_state: Record<string, unknown>;
  current_agent_idx: number;
  action_mask: boolean[];
  action_log: ActionLogEntry[];
  game_over: boolean;
  winner_idx: number | null;
  eliminated_indices: number[];
  players: PlayerInfo[];
}
