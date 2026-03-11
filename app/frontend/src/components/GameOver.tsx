import type { PlayerInfo } from "../types";

interface GameOverProps {
  players: PlayerInfo[];
  winnerIdx: number | null;
  eliminatedIndices: number[];
  playerName: string;
  onPlayAgain: () => void;
}

export default function GameOver({
  players,
  winnerIdx,
  eliminatedIndices,
  playerName,
  onPlayAgain,
}: GameOverProps) {
  // Sort: winner first, then by prestige desc, eliminated last
  const sorted = [...players].sort((a, b) => {
    if (a.player_idx === winnerIdx) return -1;
    if (b.player_idx === winnerIdx) return 1;
    const aElim = eliminatedIndices.includes(a.player_idx);
    const bElim = eliminatedIndices.includes(b.player_idx);
    if (aElim && !bElim) return 1;
    if (!aElim && bElim) return -1;
    return b.total_prestige - a.total_prestige;
  });

  return (
    <div className="game-over">
      <h2>Game Over</h2>
      <pre style={{ color: "var(--fg-dim)", fontSize: 12 }}>
{`  ┌─────────────────┐
  │   FINAL SCORES   │
  └─────────────────┘`}
      </pre>
      <div className="game-over-results">
        {sorted.map((p) => {
          const isWinner = p.player_idx === winnerIdx;
          const isElim = eliminatedIndices.includes(p.player_idx);
          const name = p.is_human ? playerName : p.player_name;

          let cls = "result-row";
          if (isWinner) cls += " winner";
          if (isElim) cls += " eliminated";

          return (
            <div key={p.player_idx} className={cls}>
              <span>
                {isWinner ? ">>> " : "    "}
                {name}
                {isElim ? " [ELIMINATED]" : ""}
                {isWinner ? " [WINNER]" : ""}
              </span>
              <span>
                Prestige: {p.total_prestige} | Money: {p.total_money}
              </span>
            </div>
          );
        })}
      </div>
      <button onClick={onPlayAgain}>Play Again</button>
    </div>
  );
}
