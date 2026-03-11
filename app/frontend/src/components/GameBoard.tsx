import type { GameResponse, ActionLogEntry } from "../types";
import AuctionCard from "./AuctionCard";
import PlayerPanel from "./PlayerPanel";
import MoneyCards from "./MoneyCards";
import ActionLog from "./ActionLog";

interface GameBoardProps {
  response: GameResponse;
  playerName: string;
  cumulativeLog: ActionLogEntry[];
  loading: boolean;
  playbackActive: boolean;
  activeRobotName: string | null;
  onPlay: (action: number) => void;
}

export default function GameBoard({
  response,
  playerName,
  cumulativeLog,
  loading,
  playbackActive,
  activeRobotName,
  onPlay,
}: GameBoardProps) {
  const { players, action_mask, current_agent_idx, game_state } = response;
  const humanPlayer = players.find((p) => p.is_human)!;

  // Extract auction card from game_state
  const curRound = game_state.cur_round as {
    card: { type: string; value: number | null; speciality: string | null };
    cur_bid: number;
    cards_in_bid: Record<string, number[]>;
  };
  const auctionCard = curRound.card;
  const currentBid = curRound.cur_bid;
  const remainingSpecial = game_state.remaining_special_cards as number;

  // Cards the human currently has in their bid
  const humanBidCards: number[] = curRound.cards_in_bid["0"] ?? [];

  // Determine the auction leader (highest bid among active players)
  const leaderIdx = (() => {
    let maxBid = 0;
    let leader: number | null = null;
    for (const p of players) {
      if (p.is_active_in_auction && p.current_bid > maxBid) {
        maxBid = p.current_bid;
        leader = p.player_idx;
      }
    }
    return leader;
  })();

  const inputDisabled = loading || playbackActive;

  return (
    <div className="game-board">
      <div className="game-main">
        <AuctionCard
          card={auctionCard}
          currentBid={currentBid}
          remainingSpecial={remainingSpecial}
        />
        <MoneyCards
          moneyCards={humanPlayer.money_cards}
          actionMask={action_mask}
          cardsInBid={humanBidCards}
          onPlay={onPlay}
          disabled={inputDisabled}
        />
      </div>
      <div className="game-sidebar">
        {players.map((p) => (
          <PlayerPanel
            key={p.player_idx}
            player={p}
            isCurrent={p.player_idx === current_agent_idx}
            isEliminated={false}
            isWinner={false}
            isLeader={p.player_idx === leaderIdx}
            isPlayingNow={
              playbackActive && p.player_name === activeRobotName
            }
            playerName={playerName}
          />
        ))}
        <ActionLog entries={cumulativeLog} playbackActive={playbackActive} />
      </div>
    </div>
  );
}
