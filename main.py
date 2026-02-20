"""Main benchmark runner script."""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from src.game_generator import generate_game_batch
from src.llm_interface import DummyLLM
from src.benchmark import Benchmark


def run_benchmark(
    num_games: int = 100,
    num_rows: int = 3,
    num_cols: int = 3,
    payoff_range: tuple = (-100, 100),
    seed: int = None,
    output_dir: str = "results",
    llm_seed: int = None
):
    """
    Run the benchmark.
    
    Args:
        num_games: Number of games to evaluate
        num_rows: Number of row player actions
        num_cols: Number of column player actions
        payoff_range: Range of payoff values
        seed: Seed for game generation
        output_dir: Directory to save results
        llm_seed: Seed for LLM randomness
    """
    print(f"Generating {num_games} games ({num_rows}x{num_cols})...")
    games = generate_game_batch(num_games, num_rows, num_cols, payoff_range, seed=seed)
    
    print("Initializing LLM interface (using dummy LLM for testing)...")
    llm = DummyLLM(seed=llm_seed, use_pure_actions=True)
    
    print("Running benchmark...")
    benchmark = Benchmark(llm)
    results, summary = benchmark.evaluate_games(games, verbose=True)
    
    # Save results
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = Path(output_dir) / f"benchmark_results_{timestamp}.json"
    summary_file = Path(output_dir) / f"benchmark_summary_{timestamp}.json"
    
    # Convert results to JSON-serializable format
    results_list = [r.to_dict() for r in results]
    
    with open(results_file, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print(f"Summary saved to {summary_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Number of games: {summary['num_games']}")
    print(f"Mean Nash gap: {summary['mean_nash_gap']:.4f}")
    print(f"Median Nash gap: {summary['median_nash_gap']:.4f}")
    print(f"Std Nash gap: {summary['std_nash_gap']:.4f}")
    print(f"Min Nash gap: {summary['min_nash_gap']:.4f}")
    print(f"Max Nash gap: {summary['max_nash_gap']:.4f}")
    print(f"Mean LLM value: {summary['mean_llm_value']:.4f}")
    print(f"Mean BR value: {summary['mean_br_value']:.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM game theory benchmark")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games")
    parser.add_argument("--num-rows", type=int, default=3, help="Number of row actions")
    parser.add_argument("--num-cols", type=int, default=3, help="Number of column actions")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--llm-seed", type=int, default=None, help="LLM randomness seed")
    
    args = parser.parse_args()
    
    run_benchmark(
        num_games=args.num_games,
        num_rows=args.num_rows,
        num_cols=args.num_cols,
        seed=args.seed,
        output_dir=args.output_dir,
        llm_seed=args.llm_seed
    )
