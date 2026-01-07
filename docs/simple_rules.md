# High Society - Simplified Rules

## Overview

High Society is an auction game by Reiner Knizia where players bid to acquire prestige cards while managing their limited money. The catch: the player with the least money at the end is eliminated from winning, regardless of their prestige!

## Full Game Rules (Original)

### Setup
- Each player gets 11 money cards with various denominations
- 16 status cards are shuffled into a draw pile:
  - 10 luxury cards worth 1-10 prestige points
  - 3 prestige multiplier cards (2x)
  - 3 disgrace cards (faux pas, scandal, passée)

### Bidding on Luxury/Prestige Cards
- Players bid sequentially by playing money cards face-up
- Each bid must be higher than the previous
- Players can pass and retrieve all their bid money
- Once you pass, you're out of that auction
- Last player standing wins the card and pays their bid

### Bidding on Disgrace Cards (Reverse Auction)
- When a disgrace card appears, the auction reverses
- First player to pass RECEIVES the card and gets their money back
- All other players lose their bid money

### Game End
- Game ends when the 4th special (green-backed) card is revealed
- No bidding on that card

### Scoring
1. Players with the LEAST money are eliminated (score 0)
2. Among remaining players:
   - Sum luxury card values
   - Apply multipliers from prestige cards
   - Subtract disgrace penalties
3. Highest prestige wins!

---

## Simplified Version (For RL Project)

### Key Simplifications

**1. No Denominations**
- Each player starts with a total of 45 money (sum of 1-9)
- Players bid by announcing amounts directly
- No physical money cards to manage

**2. No Disgrace Cards**
- Only positive prestige cards in the deck:
  - 9 value cards (worth 1-9 prestige)
  - 4 special cards (2x multipliers)
- Total: 13 cards

**3. Simplified Bidding**
- Bid 0 to pass
- Must bid at least current_bid + 1
- Bids are deducted from your total money
- When you pass, your bid money is returned

### Setup
- 3+ players
- Each player starts with 45 money
- Shuffle all 13 prestige cards into a deck

### Game Flow

**Each Auction Round:**
1. Draw the top card from the deck
2. Starting with the round starter, players bid sequentially
3. Each player must either:
   - **Bid**: Announce amount ≥ (current_bid + 1)
   - **Pass**: Bid 0, get money back, exit auction
4. Continue until only one player remains (or all pass)
5. Winner gets the card and pays their bid
6. Round starter advances to next player

**Turn Order:**
- Round starter rotates clockwise each round
- Within a round, bidding proceeds clockwise
- Players who pass are skipped for rest of that round

### Game End
- Game ends when all 4 special (2x) cards have been drawn
- **Important**: No bidding on the 4th special card - game ends immediately

### Scoring
1. Calculate each player's remaining money
2. **Eliminate** player(s) with the LEAST money (they score 0)
3. For remaining players, calculate prestige:
   - Sum all value cards
   - Multiply by 2 for each 2x card owned
4. Highest prestige wins!

### Example Scoring
```
Player A: Cards [5, 3, 2x], Money: 10
Player B: Cards [9, 7], Money: 15
Player C: Cards [6, 4, 2x, 2x], Money: 8

Player C eliminated (least money)
Player A: (5 + 3) × 2 = 16 prestige
Player B: 9 + 7 = 16 prestige (tie!)
Player C: 0 (eliminated)
```

### Strategy Considerations
- **Money management**: Don't get eliminated! Save enough money
- **Multiplier timing**: 2x cards are most valuable with high-value cards
- **Opponent tracking**: Watch others' money and prestige
- **Round counting**: Only 13 rounds total, 4 special cards end the game

---

## Differences from Full Game

| Aspect | Full Game | Simplified |
|--------|-----------|------------|
| Money | 11 cards with denominations | Single pool of 45 |
| Bidding | Play specific cards | Announce amounts |
| Disgrace cards | Yes (reverse auctions) | No |
| Total cards | 16 | 13 |
| Game end | 4th special card revealed | 4th special card drawn |
| Prestige cards | 1-10 value | 1-9 value |

## Implementation Notes

The simplified version maintains the core strategic tension of the original:
- Auction dynamics (when to bid, when to fold)
- Resource management (money vs prestige)
- Elimination rule creates interesting risk/reward
- Multipliers add strategic depth

But removes complexity that's harder for RL agents to learn:
- Combinatorial bidding with denominations
- Asymmetric disgrace card mechanics
- Slightly shorter games (13 vs 16 cards)
