# LLM Game Theory Benchmark

Measure how well a raw LLM plays in simple strategic settings before any post-training.

## Project Overview

This project implements a benchmark for evaluating LLM performance in zero-sum matrix games. The benchmark computes the "Nash gap" — the difference between what the LLM achieves and the best response payoff against a Nash equilibrium opponent.

### Key Concepts

- **Matrix Game**: A two-player zero-sum game with finite action spaces represented as a matrix
- **Nash Equilibrium**: A pair of mixed strategies where neither player can improve by unilateral deviation
- **Nash Gap**: The difference between best response value and LLM's value against Nash opponent
  - Gap = BR_value - LLM_value ≥ 0
  - Gap = 0 means LLM plays optimally
  - Larger gaps indicate worse performance

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── game_generator.py      # Generate random matrix games
│   ├── nash_solver.py         # Compute Nash equilibrium using LP
│   ├── llm_interface.py       # LLM query interface and response parsing
│   └── benchmark.py           # Benchmark runner and metrics
├── tests/                      # Unit tests (coming soon)
├── notebooks/                  # Jupyter notebooks for analysis
├── main.py                     # Main benchmark runner
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Create a Python environment (Python 3.8+)
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install Ollama for local LLM inference:
   - Download from https://ollama.ai
   - Pull a model: `ollama pull llama3.1`
   - Start server: `ollama serve` (in a separate terminal)

## Quick Start

### Using Ollama (Local LLM)

Start Ollama in a separate terminal:
```bash
ollama serve
```

Then run the benchmark (uses llama3.1 by default):
```bash
python main.py --num-games 10 --num-trials 10 --seed 42
```

### Using Dummy LLM (Random baseline)

For testing without Ollama:
```bash
python main.py --num-games 10 --num-trials 10 --llm-type dummy --seed 42
```

### Command Line Options

- `--num-games`: Number of games to generate (default: 100)
- `--num-trials`: Number of trials per game (default: 100)
- `--num-rows`: Number of row player actions (default: 3)
- `--num-cols`: Number of column player actions (default: 3)
- `--seed`: Random seed for game generation
- `--llm-seed`: Random seed for LLM (DummyLLM only)
- `--llm-type`: LLM backend type: `ollama`, `dummy`, `together`, `openai` (default: `dummy`)
- `--llm-model`: Specific model name (e.g., `llama3.1`, `llama2`)
- `--output-dir`: Directory to save results (default: "results")
- `--parallel`: Enable parallel workers for faster execution (especially useful for network-based LLMs)
- `--num-workers`: Number of parallel workers (default: 4, max recommended: 8-16)

### Examples

Run 100 games × 100 trials with ollama:
```bash
python main.py --num-games 100 --num-trials 100 --seed 42
```

Run smaller experiment with dummy LLM (for testing):
```bash
python main.py --num-games 10 --num-trials 10 --llm-type dummy --seed 42
```

Custom game size (2x2 games) with ollama:
```bash
python main.py --num-games 50 --num-trials 50 --num-rows 2 --num-cols 2 --seed 123
```

**Parallel execution (4-10x faster for network-based LLMs):**
```bash
# Uses 4 parallel workers (adjust with --num-workers)
python main.py --num-games 100 --num-trials 100 --llm-type together --parallel --num-workers 8 --seed 42
```

**Faster execution tips:**
- **Reduce trials:** `--num-trials 20` (5x faster)
- **Use parallelization:** `--parallel --num-workers 8` (best for remote LLMs)
- **Smaller local model:** `--llm-type ollama --llm-model llama2` (3x faster)
- **Together AI:** `--llm-type together --parallel` (good balance of speed & accuracy)

## Benchmark Protocol

For each game G with payoff matrix U:

1. **Generate Game**: Create random payoff matrix
2. **Query LLM**: Prompt LLM with game and ask for action (pure or mixed)
3. **Compute Nash**: Calculate Nash equilibrium (π₁*, π₂*)
4. **Fix Opponent**: Opponent plays Nash strategy π₂*
5. **Measure Gap**: Compute gap(G) = V_BR(G) - V_LLM(G)

### Metrics

For each game, the benchmark computes:

- **LLM Value**: V_LLM(G) = E[a ~ π_LLM, b ~ π₂*][U(a,b)]
- **Best Response Value**: V_BR(G) = max_a E[b ~ π₂*][U(a,b)]
- **Nash Gap**: gap(G) = V_BR(G) - V_LLM(G) ≥ 0

Summary statistics across all games:
- Mean, median, std, min, max Nash gap
- Mean LLM and BR values
- Gap ratio (normalized by BR value)

## Output

The benchmark generates three files in the `results/` directory:

1. **games_{timestamp}.json** - Game mapping data (100 entries)
   - Maps game_id to payoff matrix and Nash equilibria
   - Use this to look up the actual game for any trial result

2. **trials_{timestamp}.json** - Trial results (10,000 entries for 100 games × 100 trials)
   - Contains: game_id, trial_id, llm_decision, llm_value, best_response_value, nash_gap
   - Use this to analyze LLM strategy patterns

3. **summary_{timestamp}.json** - Summary statistics
   - Overall performance metrics

### Example Output Format

**Game entry:**
```json
{
  "game_id": 0,
  "payoff_matrix": [[1.5, -2.3], [-0.8, 3.1]],
  "nash_equilibrium_row": [0.45, 0.55],
  "nash_equilibrium_col": [0.60, 0.40]
}
```

**Trial entry:**
```json
{
  "game_id": 0,
  "trial_id": 0,
  "llm_decision": 1,
  "llm_value": -0.42,
  "best_response_value": 0.15,
  "nash_gap": 0.57
}
```

**Summary entry:**
```json
{
  "num_games": 100,
  "num_trials_per_game": 100,
  "total_trials": 10000,
  "mean_nash_gap": 23.45,
  "median_nash_gap": 20.12,
  "std_nash_gap": 15.67,
  "min_nash_gap": 0.05,
  "max_nash_gap": 67.89,
  "mean_llm_value": -5.34,
  "mean_br_value": 18.11
}
```

## Using Different LLM Backends

The benchmark supports multiple LLM backends:

### 1. Ollama (Local Inference) - **Recommended**
```bash
# Start Ollama (in separate terminal)
ollama serve

# Run benchmark with llama3.1
python main.py --num-games 100 --num-trials 100 --seed 42
```
- **Pros**: Free, local, fast, no API keys
- **Cons**: Requires GPU/good CPU
- **Models**: Supports any Ollama model (llama3.1, llama2, etc.)

### 2. DummyLLM (Random Baseline)
```bash
python main.py --llm-type dummy --num-games 100 --num-trials 100 --seed 42
```
- **Pros**: No dependencies, fast testing
- **Cons**: Random decisions, not a real LLM

### 3. Together AI (Hosted Llama 3)
```bash
export TOGETHER_API_KEY="your_key_here"
python main.py --llm-type together --num-games 100 --num-trials 100 --seed 42
```
- **Pros**: Hosted inference, no local GPU needed
- **Cons**: API calls cost money (~$0.001 per request)
- **Setup**: Get API key from https://www.together.ai

### 4. OpenAI (GPT-4 / GPT-3.5)
```bash
export OPENAI_API_KEY="your_key_here"
python main.py --llm-type openai --llm-model gpt-3.5-turbo --seed 42
```
- **Pros**: State-of-the-art models
- **Cons**: Costs per request (~$0.001-0.01 per game)
- **Setup**: Get API key from https://platform.openai.com

## Testing

Run unit tests:
```bash
PYTHONPATH=. python tests/test_core.py
```

## Notes

- Nash equilibrium computation uses linear programming (scipy.optimize.linprog)
- Supports both pure action and mixed strategy LLM outputs
- Results are deterministic when seeds are fixed
- Response parsing is basic; you may need to customize for specific LLMs

## Future Work

- [ ] Improved response parsing for complex LLM outputs

- [ ] More sophisticated response parsing
- [ ] Analysis and visualization notebooks
- [ ] Support for non-zero-sum games
- [ ] Batch game generation with specific properties
- [ ] Sensitivity analysis

## References

- Zero-sum games and Nash equilibrium: https://en.wikipedia.org/wiki/Zero-sum_game
- Mixed strategy Nash equilibrium: https://en.wikipedia.org/wiki/Nash_equilibrium

## License

TODO: Add license information
