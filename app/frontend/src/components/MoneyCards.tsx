interface MoneyCardsProps {
  moneyCards: number[];
  actionMask: boolean[];
  cardsInBid: number[];
  onPlay: (action: number) => void;
  disabled: boolean;
}

export default function MoneyCards({
  moneyCards,
  actionMask,
  cardsInBid,
  onPlay,
  disabled,
}: MoneyCardsProps) {
  // All possible card values 1-10
  const allCards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

  return (
    <div className="money-cards-section">
      <div className="money-cards-header">Your Hand</div>
      <div className="money-cards-hand">
        {allCards.map((val) => {
          const owned = moneyCards.includes(val);
          const inBid = cardsInBid.includes(val);
          const valid = actionMask[val] === true;
          const spent = !owned && !inBid;

          let cls = "money-card";
          if (valid && !disabled) cls += " valid";
          if (inBid) cls += " in-bid";
          if (spent) cls += " spent";

          return (
            <div
              key={val}
              className={cls}
              onClick={() => {
                if (valid && !disabled) onPlay(val);
              }}
              title={
                inBid
                  ? `${val} (in current bid)`
                  : valid
                    ? `Add ${val} to bid`
                    : spent
                      ? `${val} (spent)`
                      : `${val} (can't play)`
              }
            >
              {val}
            </div>
          );
        })}
      </div>
      <button
        className={`pass-button${actionMask[0] && !disabled ? " valid" : ""}`}
        disabled={disabled || !actionMask[0]}
        onClick={() => onPlay(0)}
      >
        Pass
      </button>
    </div>
  );
}
