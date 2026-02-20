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

## Quick Start

Run a simple benchmark with 100 random games:

```bash
python main.py --num-games 100 --seed 42
```

### Command Line Options

- `--num-games`: Number of games to benchmark (default: 100)
- `--num-rows`: Number of row player actions (default: 3)
- `--num-cols`: Number of column player actions (default: 3)
- `--seed`: Random seed for reproducibility
- `--output-dir`: Directory to save results (default: "results")
- `--llm-seed`: Seed for LLM randomness

### Example

```bash
python main.py --num-games 500 --num-rows 4 --num-cols 4 --seed 123 --output-dir results/exp1
```

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

Results are saved to the `results/` directory:

- `benchmark_results_{timestamp}.json`: Detailed results for each game
- `benchmark_summary_{timestamp}.json`: Summary statistics

### Example Output Format

```json
{
  "num_games": 100,
  "mean_nash_gap": 23.45,
  "median_nash_gap": 20.12,
  "std_nash_gap": 15.67,
  "min_nash_gap": 0.05,
  "max_nash_gap": 67.89,
  "mean_llm_value": -5.34,
  "mean_br_value": 18.11
}
```

## Using with Real LLMs

To use with a real LLM (e.g., OpenAI API):

1. Create a custom LLMInterface subclass:
```python
from src.llm_interface import LLMInterface

class OpenAILLM(LLMInterface):
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
    
    def query(self, game: MatrixGame) -> Union[int, np.ndarray]:
        prompt = GamePromptFormatter.format_game_as_text(game)
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        # Parse response...
```

2. Use in benchmark:
```python
llm = OpenAILLM()
benchmark = Benchmark(llm)
results, summary = benchmark.evaluate_games(games)
```

## Testing

Run tests (coming soon):
```bash
pytest tests/
```

## Notes

- Currently uses a dummy LLM for testing. Replace with actual LLM implementation.
- Nash equilibrium computation uses linear programming (scipy.optimize.linprog)
- Supports both pure action and mixed strategy LLM outputs
- Results are deterministic when seeds are fixed

## Future Work

- [ ] Integration with actual LLM APIs
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
