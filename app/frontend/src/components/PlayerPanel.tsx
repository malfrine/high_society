import type { PlayerInfo } from "../types";

interface PlayerPanelProps {
  player: PlayerInfo;
  isCurrent: boolean;
  isEliminated: boolean;
  isWinner: boolean;
  isLeader: boolean;
  isPlayingNow: boolean;
  playerName: string;
}

export default function PlayerPanel({
  player,
  isCurrent,
  isEliminated,
  isWinner,
  isLeader,
  isPlayingNow,
  playerName,
}: PlayerPanelProps) {
  const displayName = player.is_human ? playerName : player.player_name;

  let cls = "player-panel";
  if (player.is_human) cls += " is-human";
  if (isCurrent) cls += " is-current";
  if (isEliminated) cls += " eliminated";
  if (isLeader) cls += " is-leader";
  if (isPlayingNow) cls += " is-playing-now";

  return (
    <div className={cls}>
      <div className="player-name">
        {displayName}
        {isWinner ? " [WINNER]" : ""}
        {isLeader ? " [LEADER]" : ""}
        {player.is_human ? " (you)" : ""}
      </div>
      <div className="player-stats">
        <span className="prestige">Prestige: {player.total_prestige}</span>
        <span className="money">Money: {player.total_money}</span>
      </div>
      <div className="player-bid">
        {player.is_active_in_auction ? (
          player.current_bid > 0 ? `Bid: ${player.current_bid}` : "No bid yet"
        ) : (
          <span className="player-inactive">Passed</span>
        )}
      </div>
    </div>
  );
}
