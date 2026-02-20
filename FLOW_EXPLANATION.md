# Complete Dataset & Benchmark Flow Explanation

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GAME GENERATION (One-time)                          │
├─────────────────────────────────────────────────────────────────────────┤
│  • Create 100 random 3×3 zero-sum games                                │
│  • Each game has payoff matrix U[i,j] = Row player's payoff            │
│  • Column player's payoff = -U[i,j] (zero-sum constraint)              │
│  • Payoffs randomly sampled from [-100, 100]                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│           GAME SETUP: Compute Nash Equilibrium (One-time)               │
├─────────────────────────────────────────────────────────────────────────┤
│  For each game, solve 2 linear programs:                               │
│                                                                         │
│  1. ROW PLAYER'S STRATEGY (Nash mixed strategy σ_r):                   │
│     • Maximize minimum expected payoff                                  │
│     • Formula: max_σ min_j Σ_i U[i,j] * σ_r[i]                       │
│     • Result: σ_r = [p₁, p₂, p₃] (probability distribution)           │
│                                                                         │
│  2. COLUMN PLAYER'S STRATEGY (Nash mixed strategy σ_c):                │
│     • Minimize maximum expected loss                                    │
│     • Formula: min_σ max_i Σ_j U[i,j] * σ_c[j]                       │
│     • Result: σ_c = [q₁, q₂, q₃] (probability distribution)           │
│                                                                         │
│  3. NASH VALUE: v* = σ_r^T @ U @ σ_c                                  │
│     • Expected payoff when both play Nash equilibrium                   │
│                                                                         │
│  Key insight: Column player ALWAYS plays σ_c (fixed!)                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│        DATASET STRUCTURE: games.json (Saved once per experiment)        │
├─────────────────────────────────────────────────────────────────────────┤
│  [                                                                       │
│    {                                                                     │
│      "game_id": 0,                                                      │
│      "payoff_matrix": [[-25.09, 90.14, 46.40],                          │
│                        [19.73, -68.80, -68.80],                        │
│                        [-88.38, 73.24, 20.22]],                        │
│      "nash_equilibrium_row": [0.553, 0.447, 0.0],    ← Row plays this  │
│      "nash_equilibrium_col": [0.720, 0.0, 0.280]     ← Column plays this│
│    },                                                                    │
│    ...100 games total...                                                │
│  ]                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│      BENCHMARK TRIAL LOOP: For each trial of each game                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  📝 STEP 1: QUERY LLM                                                   │
│  ───────────────────────                                                │
│  Input:  Game matrix (formatted as text)                               │
│  Output: LLM chooses ONE action: 0, 1, or 2                            │
│                                                                         │
│  Example prompt:                                                        │
│  "You're the row player. Payoff matrix:                                │
│   Action 0: [-25.09,  90.14,  46.40]                                   │
│   Action 1: [19.73,  -68.80, -68.80]                                   │
│   Action 2: [-88.38,  73.24,  20.22]                                   │
│   Choose action 0, 1, or 2."                                            │
│                                                                         │
│  LLM response: "I choose action 0"                                      │
│  Parsed: llm_decision = 0                                               │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  ⚔️  STEP 2: GAME OUTCOME - LLM vs Nash Opponent                        │
│  ──────────────────────────────────────────────────                     │
│  • LLM chose action: 0                                                  │
│  • Opponent plays Nash strategy: σ_c = [0.720, 0.0, 0.280]             │
│                                                                         │
│  LLM's strategy (pure action 0):                                        │
│    σ_llm = [1.0, 0.0, 0.0]  (100% probability on action 0)            │
│                                                                         │
│  Expected payoff for LLM:                                               │
│    LLM_value = σ_llm @ U @ σ_c                                          │
│               = [1.0, 0.0, 0.0] @ U @ [0.720, 0.0, 0.280]             │
│               = U[0,:] @ σ_c                                            │
│               = [-25.09*0.720 + 90.14*0.0 + 46.40*0.280]               │
│               = -18.06 + 0 + 12.99                                      │
│               = -5.07  ← What LLM got                                   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  🏆 STEP 3: BEST RESPONSE COMPUTATION                                    │
│  ──────────────────────────────                                         │
│  What could LLM have gotten if it played OPTIMALLY against σ_c?        │
│                                                                         │
│  Compute payoff for each possible action against σ_c:                  │
│    BR[0] = U[0,:] @ σ_c = -18.06 + 0 + 12.99 = -5.07                  │
│    BR[1] = U[1,:] @ σ_c = 19.73*0.720 - 68.80*0.0 - 68.80*0.280       │
│           = 14.20 + 0 - 19.26 = -5.06                                  │
│    BR[2] = U[2,:] @ σ_c = -88.38*0.720 + 73.24*0.0 + 20.22*0.280      │
│           = -63.63 + 0 + 5.66 = -57.97                                 │
│                                                                         │
│  Best response action: argmax{-5.07, -5.06, -57.97} = 1               │
│  Best response value: -5.06  ← What LLM should have gotten             │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  📊 STEP 4: COMPUTE NASH GAP METRIC                                     │
│  ────────────────────────────────────                                   │
│  Nash gap = Best Response Value - LLM Value                            │
│          = (-5.06) - (-5.07)                                            │
│          = 0.01                                                         │
│                                                                         │
│  Interpretation:                                                        │
│    • Nash gap = 0: LLM played optimally against Nash opponent          │
│    • Nash gap > 0: LLM played suboptimally (lost money)                │
│    • Nash gap = 50: LLM could have earned 50 more by playing BR       │
│                                                                         │
│  For 100 games × 1 trial (what we ran):                                │
│    • Mean gap: 17.59 (on average, LLM suboptimal by 17.59)             │
│    • Median gap: ~0 (half the games played optimally)                  │
│    • Hard games: 16 games with gap > 50 (LLM very suboptimal)          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│      DATASET STRUCTURE: trials.json (Saved after all trials)            │
├─────────────────────────────────────────────────────────────────────────┤
│  [                                                                       │
│    {                                                                     │
│      "game_id": 0,                                                      │
│      "trial_id": 0,                                                     │
│      "llm_decision": 0,              ← Action chosen by LLM             │
│      "llm_value": -5.067,            ← Payoff LLM achieved              │
│      "best_response_value": -5.067,  ← Best possible payoff             │
│      "nash_gap": 0.0                 ← Difference (optimality metric)   │
│    },                                                                    │
│    {                                                                     │
│      "game_id": 1,                                                      │
│      "trial_id": 0,                                                     │
│      "llm_decision": 0,              ← LLM chose action 0               │
│      "llm_value": -78.828,           ← But got -78.828 (bad!)           │
│      "best_response_value": -42.149, ← Could have gotten -42.149        │
│      "nash_gap": 36.679              ← Lost 36.679 by poor choice       │
│    },                                                                    │
│    ...100 trials total (1 per game)...                                 │
│  ]                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Conceptual Points

### 1. **The Opponent is ALWAYS Playing Nash**
- Column player strategy σ_c is computed ONCE during setup
- **Same σ_c is used for ALL trials of that game**
- LLM is measured against this fixed, optimal opponent
- This is intentional: we want to measure LLM performance against best-play

### 2. **LLM Gets ONE Choice Per Trial**
- LLM sees the game matrix and chooses a pure action (0, 1, or 2)
- No mixing: LLM cannot randomize
- For multiple trials, we query the LLM multiple times (gets different answers)
- This tests LLM consistency and robustness

### 3. **The Comparison Logic**
```
LLM's payoff:           σ_llm @ U @ σ_c  (vector @ matrix @ vector)
Best response payoff:   max_i (U[i,:] @ σ_c)  (best single action vs Nash)

Nash gap = BR payoff - LLM payoff

If gap = 0:   LLM played optimally
If gap > 0:   LLM was suboptimal by this amount
If gap = 100: LLM could have earned 100 more points
```

### 4. **Why We Use Nash Column Strategy**
- **Test hypothesis:** "Can LLM play game-theoretically sound strategies?"
- **Measuring against Nash** = measuring against best-possible opponent
- This isolates LLM's strategic understanding from opponent behavior
- If opponent plays Nash, LLM CANNOT do better than its best response

### 5. **The Three Values Explained**

| Value | What It Is | How Computed |
|-------|-----------|--------------|
| `llm_value` | What the LLM achieved | LLM's action vector @ payoff matrix @ Nash column strategy |
| `best_response_value` | Best possible against Nash | max over all actions of (action @ payoff matrix @ Nash strategy) |
| `nash_gap` | Suboptimality metric | Best response value - LLM value |

---

## Complete Example (Game 1)

```
Game Matrix U:
  Col0   Col1   Col2
  ─────────────────────
  19.73  -68.80 -68.80  ← Action 0
  19.73  -68.80 -68.80  ← Action 1  
  -88.38  73.24  20.22  ← Action 2

Nash Equilibrium (computed once):
  Row player should play: σ_r = [some mixture]
  Column player should play: σ_c = [0.72, 0.0, 0.28]

Trial 0:
  LLM is shown the matrix
  LLM response: "I choose action 0"
  
  Computation:
    LLM strategy:           σ_llm = [1.0, 0.0, 0.0]
    LLM value:              [1.0, 0.0, 0.0] @ U @ [0.72, 0.0, 0.28]
                          = 19.73*0.72 - 68.80*0.0 - 68.80*0.28
                          = 14.21 - 19.26 = -5.05
    
    Best response payoffs:
      BR[0] = 19.73*0.72 - 68.80*0 - 68.80*0.28 = -5.05
      BR[1] = 19.73*0.72 - 68.80*0 - 68.80*0.28 = -5.05
      BR[2] = -88.38*0.72 + 73.24*0 + 20.22*0.28 = -57.97
    
    Best response value = max{-5.05, -5.05, -57.97} = -5.05
    Nash gap = -5.05 - (-5.05) = 0.0 ✓ (LLM played optimally!)

Trial 1 (same game, different LLM response):
  LLM is shown the same matrix again
  LLM response: "I choose action 2"  ← Different answer!
  
  Computation:
    LLM strategy:           σ_llm = [0.0, 0.0, 1.0]
    LLM value:              [0.0, 0.0, 1.0] @ U @ [0.72, 0.0, 0.28]
                          = -88.38*0.72 + 73.24*0 + 20.22*0.28
                          = -63.63 + 0 + 5.66 = -57.97
    
    Best response value = -5.05 (same as before)
    Nash gap = -5.05 - (-57.97) = 52.92 ✗ (LLM played poorly!)
```

---

## Summary

**The complete flow:**
1. **Generate** 100 random 3×3 games
2. **Setup**: For each game, compute the Nash equilibrium (pure strategy pairs or mixed strategies)
3. **Key decision**: Column player ALWAYS plays its Nash strategy σ_c
4. **Each trial**: Query LLM, get one action, compute payoff against σ_c
5. **Measure**: Compare LLM's payoff to what LLM could have achieved (best response)
6. **Nash gap** quantifies: How many points LLM lost by not playing optimally

**The key insight:**
- By fixing column player to Nash strategy, we isolate pure strategic acumen
- LLM cannot "beat" the opponent through opponent mistakes
- Nash gap measures true game-theoretic understanding
- High gap = LLM doesn't understand the strategic structure of the game
