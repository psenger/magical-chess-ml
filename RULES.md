# Magical Chess Rules - Call of Antia Microgame

**Magical Chess** (also called **Magic Chess** or **Merge Magic**) is a chess-themed merge puzzle mini-game in the mobile game *Call of Antia: Match 3 RPG*. It is an event-based drag-and-drop merge game where players combine identical chess pieces to create higher-level pieces and earn points. Scores contribute to event rewards like heroes, gems, and resources. The game uses limited **Magical Hourglasses** as energy.

## Board and Setup

- **Board Size**: 4x4 grid
- **Initial Board**: Random placement of low-level pieces
- **Movement**: **Drag-and-drop only** - there is NO swipe/slide mechanic like 2048
  - Manually drag a piece and drop it onto matching pieces to trigger a merge
  - Pieces stay in place until you move them

## Chess Pieces and Tiers

Pieces are visually represented as chess figures with tier colors (Bronze/Silver/Gold):

| Level | Chess Piece | Tier   | Display       |
|-------|-------------|--------|---------------|
| 1     | Pawn        | Bronze | B1            |
| 2     | Knight      | Bronze | B2            |
| 3     | Bishop      | Bronze | B3            |
| 4     | Rook        | Silver | S4            |
| 5     | Queen       | Silver | S5            |
| 6     | King        | Silver | S6            |
| 7+    | Super King  | Gold   | G7, G8, G9... |

## Merge Rules

### Merge-3 (Minimum)
- **3 identical pieces → 1 piece of the next level**
- Example: B1 + B1 + B1 = B2
- Pieces must be the **same level AND same tier**

### Merge-5 (Optimal)
- **5 identical pieces → 2 pieces of next level**
- Example: B1 + B1 + B1 + B1 + B1 = B2 + B2
- This is the most efficient merge – always aim for Merge-5

### Merge-4 (Inefficient)
- Some versions allow 4-merge, but it only yields one piece of the next level
- Wastes the fourth piece – avoid this

### Merge-2 (Invalid)
- **2 pieces do NOT merge** - minimum is 3
- Two adjacent identical pieces will stay separate

### Tier Restrictions
- **Same-tier only**: Bronze merges with Bronze, Silver with Silver, Gold with Gold
- **Same-level only**: Must be the exact same level (e.g., three B2s, not B1+B2+B2)
- B3 + B3 + B3 = S4 (transitions from Bronze to Silver)
- S6 + S6 + S6 = G7 (transitions from Silver to Gold)

## How Merging Works

1. **Drag** a piece from its tile
2. **Drop** it onto other matching pieces (same tier, same level)
3. Pieces can be **adjacent** (horizontally or vertically) or you can drag into a cluster
4. Merge-5 can **span multiple rows/columns** as long as pieces are touching
5. **Visual cue**: Pieces will "pulse" when hovering to show if a 3-merge or 5-merge will trigger
6. **Result**: Merged pieces vanish, new piece(s) appear at drop location

## Multipliers and Hourglass Costs

Before spawning a new piece, select a multiplier:

| Multiplier | New Piece Level | Hourglass Cost | Notes                            |
|------------|-----------------|----------------|----------------------------------|
| 1x         | B1 or B2        | 0-1            | Slow progress, more plays        |
| 4x         | B3              | 4              | Mid-game builder                 |
| 16x        | S5              | 16             | End-game power, adds high pieces |

- Multipliers also **scale points** from merges
- Hourglasses can be earned, bought, or used to extend games

> **ML Implementation Note**: For our ML/RL solution, multipliers are **not included** in the action space. The system assumes a fixed **1x multiplier** for all moves. This simplifies the model to focus on merge strategy while optimizing for hourglass efficiency.

## Scoring

- **Points**: Earned from merges (higher levels = more points)
- **Merge-5 bonus**: More efficient point generation
- **High Scores**: Examples from leaderboards: 10,000+, 86,000, 139,000
- Event rewards based on total score across multiple rounds

### Reward Tiers
- **Bronze Cache**: Base tier rewards
- **Silver Coffer**: Mid-tier (requires reaching point threshold with Level 4+ pieces)
- **Gold Vault**: Top tier (requires high-level merges and speed)

## Game Over

- Ends when the board is full **and no valid merges are possible**
- A valid merge requires 3+ identical same-tier/same-level pieces that can be combined
- Restart with remaining hourglasses for more rounds

## Strategies for High Scores

- **Always aim for Merge-5**: Two upgraded pieces instead of one
- **Avoid Merge-4**: Wastes a piece
- **Wait for opportunities**: Don't rush Merge-3 if you can wait for Merge-5
- **Watch for pulses**: Pieces pulse to indicate pending merge count
- **Tier transitions**: Plan for B3→S4 and S6→G7 transitions
- **Eliminate low pieces**: Prioritize merging Pawns (B1) to clear board space

## ML/Simulator Design Notes

- **State space**: 4x4 grid with piece values (tier + level) + hourglass count
- **Action space**: Source tile selection + target tile selection (drag-drop)
- **Reward**: Hourglass efficiency (score per hourglass spent)
- **Valid actions**: Only moves that result in valid merges (3+ matching adjacent pieces)
- **Stochastic elements**: Random new piece position after each action

## Event Notes

- Runs periodically (e.g., 05:00 UTC start/end, ~7 days)
- Recent 2025 iterations: January 9-16, 2025
- Redeem points for *Call of Antia* rewards

---

**Sources**: Compiled from player guides, videos, and user clarifications.
