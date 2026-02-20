#!/usr/bin/env python3
"""
Research benchmark for UAI submission.

Recommended setup for reproducible, citable research:
- Use Together AI (hosted llama3.1) for consistency
- Parallel execution (14 workers) for speed
- Multiple seeds for robustness
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime


def run_experiment(exp_name: str, num_games: int = 100, num_trials: int = 100, seed: int = 42):
    """Run a single experiment configuration."""
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"Games: {num_games}, Trials: {num_trials}, Seed: {seed}")
    print('='*60)
    
    cmd = [
        "python", "main.py",
        "--num-games", str(num_games),
        "--num-trials", str(num_trials),
        "--llm-type", "together",  # Use Together AI (hosted llama3.1)
        "--parallel",  # Already auto-enabled for together, but explicit
        "--num-workers", "14",
        "--seed", str(seed),
        "--output-dir", f"results/{exp_name}"
    ]
    
    subprocess.run(cmd, check=True)
    print(f"✓ Completed: {exp_name}")


def main():
    """Run full research benchmark suite."""
    
    # Create output directory
    base_dir = Path("results")
    base_dir.mkdir(exist_ok=True)
    
    print("LLM Game Theory Benchmark - UAI Research Suite")
    print("Using Together AI (hosted Llama 3.1) + 14 parallel workers")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Experiment 1: 2x2 games (simple)
    run_experiment("exp_2x2_games", num_games=100, num_trials=100, seed=42)
    
    # Experiment 2: 3x3 games (default)
    run_experiment("exp_3x3_games", num_games=100, num_trials=100, seed=42)
    
    # Experiment 3: 4x4 games (complex)
    run_experiment("exp_4x4_games", num_games=100, num_trials=100, seed=42)
    
    # Experiment 4: Robustness check (different seed)
    run_experiment("exp_robustness", num_games=100, num_trials=100, seed=123)
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {base_dir}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("='*60")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        print("\nMake sure you have set: export TOGETHER_API_KEY='your_key_here'")
