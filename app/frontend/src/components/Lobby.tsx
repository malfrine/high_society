import { useState } from "react";

interface LobbyProps {
  onStart: (numPlayers: number, robotType: string, playerName: string) => void;
  loading: boolean;
}

export default function Lobby({ onStart, loading }: LobbyProps) {
  const [name, setName] = useState("You");
  const [numOpponents, setNumOpponents] = useState(3);
  const [robotType, setRobotType] = useState("dqn");

  return (
    <div className="lobby">
      <h1>High Society</h1>
      <pre style={{ color: "var(--fg-dim)", fontSize: 12, textAlign: "center" }}>
{`  ┌─────────────────┐
  │  AUCTION HOUSE   │
  │   est. 2025      │
  └─────────────────┘`}
      </pre>
      <div className="lobby-form">
        <label>
          Your name
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            maxLength={20}
            placeholder="Player name"
          />
        </label>
        <label>
          Opponents
          <select
            value={numOpponents}
            onChange={(e) => setNumOpponents(Number(e.target.value))}
          >
            <option value={2}>2 opponents</option>
            <option value={3}>3 opponents</option>
            <option value={4}>4 opponents</option>
          </select>
        </label>
        <label>
          Robot type
          <select
            value={robotType}
            onChange={(e) => setRobotType(e.target.value)}
          >
            <option value="dqn">DQN (trained)</option>
            <option value="random">Random</option>
          </select>
        </label>
        <button
          onClick={() => onStart(numOpponents + 1, robotType, name)}
          disabled={loading || !name.trim()}
        >
          {loading ? "Starting..." : "New Game"}
        </button>
      </div>
    </div>
  );
}
