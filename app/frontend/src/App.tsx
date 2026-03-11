import { useGame } from "./hooks/useGame";
import Lobby from "./components/Lobby";
import GameBoard from "./components/GameBoard";
import GameOver from "./components/GameOver";

export default function App() {
  const {
    phase,
    response,
    loading,
    error,
    playerName,
    cumulativeLog,
    playbackActive,
    activeRobotName,
    startGame,
    playAction,
    resetGame,
  } = useGame();

  return (
    <>
      {loading && !playbackActive && (
        <div className="loading-overlay">
          <div className="loading-text">Thinking...</div>
        </div>
      )}

      {error && <div className="error-banner">{error}</div>}

      {phase === "lobby" && <Lobby onStart={startGame} loading={loading} />}

      {phase === "playing" && response && (
        <GameBoard
          response={response}
          playerName={playerName}
          cumulativeLog={cumulativeLog}
          loading={loading}
          playbackActive={playbackActive}
          activeRobotName={activeRobotName}
          onPlay={playAction}
        />
      )}

      {phase === "game_over" && response && (
        <GameOver
          players={response.players}
          winnerIdx={response.winner_idx}
          eliminatedIndices={response.eliminated_indices}
          playerName={playerName}
          onPlayAgain={resetGame}
        />
      )}
    </>
  );
}
