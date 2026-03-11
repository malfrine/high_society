interface AuctionCardProps {
  card: { type: string; value: number | null; speciality: string | null };
  currentBid: number;
  remainingSpecial: number;
}

export default function AuctionCard({ card, currentBid, remainingSpecial }: AuctionCardProps) {
  const isSpecial = card.type === "special";
  const display = isSpecial ? "2x" : String(card.value);
  const cls = isSpecial ? "card-special" : "card-value";

  return (
    <div className="auction-card">
      <div className="auction-card-header">Current Auction</div>
      <pre className="auction-card-ascii">
{
`┌─────────┐
│         │
│  ${display.padStart(3)}    │
│         │
└─────────┘`
}
      </pre>
      <div className={cls}>{isSpecial ? "x2 Multiplier" : `${card.value} Prestige`}</div>
      <div className="auction-info">
        Current bid: {currentBid} | Special cards left: {remainingSpecial}
      </div>
    </div>
  );
}
