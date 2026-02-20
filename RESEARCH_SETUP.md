# Research Setup Guide for UAI

This guide covers setting up the benchmark for research publication at UAI using Together AI (hosted Llama 3.1).

## Why Together AI?

For UAI submissions, we recommend **Together AI** because:

1. **Reproducibility**: Same model, deterministic endpoint
2. **Cost**: ~$0.02 per 100 games (very cheap)
3. **Speed**: 14 parallel workers = ~2-5 minutes for 100×100 benchmark
4. **Easy to cite**: Clear model name and API endpoint
5. **Research-friendly**: Academic discounts available

## Setup

### 1. Get Together AI API Key

```bash
# Sign up at https://www.together.ai
# Create account -> Get API key
export TOGETHER_API_KEY="your_api_key_here"
```

### 2. Verify Setup

```bash
# Check that key is set
echo $TOGETHER_API_KEY

# Test connection
python -c "
from src.llm_interface import TogetherAILLM
from src.game_generator import MatrixGame
import numpy as np

llm = TogetherAILLM()
game = MatrixGame(np.array([[1, -1], [-1, 1]], dtype=float))
result = llm.query(game)
print(f'Test successful! LLM returned: {result}')
"
```

### 3. Run Full Benchmark

```bash
# Single experiment (combined pure + mixed)
python main.py --num-games 100 --num-trials 100 --combined --seed 42

# Multiple experiments (for robustness)
python run_research_benchmark.py
```

## Cost Estimation

| Setup | Games | Trials | Tokens* | Cost |
|-------|-------|--------|---------|------|
| 100 | 100 | ~400K | $0.04 |
| 1000 | 100 | ~4M | $0.40 |
| 100 | 100 × 3 runs | ~1.2M | $0.12 |

*Rough estimate based on prompt + response size

## Speed on Your Computer

With 14 parallel workers + Together AI:

- **100 games × 100 trials**: ~2-5 minutes
- **100 games × 20 trials**: ~30-60 seconds
- **1000 games × 50 trials**: ~10-15 minutes

## Output Format

Combined experiments overwrite the same folder each run:

```
results/pure_and_mixed_latest/
├── games.json                  # 100 game definitions + Nash equilibria
├── trials_pure_actions.json    # Pure action trials
├── summary_pure_actions.json
├── trials_mixed_strategy.json  # Mixed strategy trials
└── summary_mixed_strategy.json
```

## For Publication

Include in methods section:

> "We benchmarked LLM game-theoretic reasoning using 100 randomly generated 3×3 zero-sum matrix games, with 100 trials per game. The LLM was queried via Together AI's hosted Llama 3.1 70B model (together.ai). Parallel execution with 14 workers ensured reproducibility and efficiency."

## Alternatives (if Together AI is unavailable)

### Groq (Free tier, extremely fast - coming soon)
```bash
# Will support when API is stable
python main.py --num-games 100 --num-trials 100 --llm-type groq --parallel --num-workers 14
```

## Troubleshooting

### "Error: TOGETHER_API_KEY not set"
```bash
export TOGETHER_API_KEY="your_key_here"
echo $TOGETHER_API_KEY  # Verify it's set
```

### Network timeouts
- Increase timeout: Edit `src/llm_interface.py` line in `TogetherAILLM.query()`
- Reduce parallel workers: `--num-workers 8`

### Rate limiting
- Together AI has generous limits (~1000 requests/min)
- If hitting limits, use: `--num-workers 4`

## Questions?

For UAI submissions with specific questions about methodology, reach out to Together AI's research team at research@together.ai.
