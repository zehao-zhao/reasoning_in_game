"""Main benchmark runner script."""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from src.game_generator import generate_game_batch
from src.llm_interface import DummyLLM, OllamaLLM, TogetherAILLM, OpenAILLM
from src.benchmark import Benchmark


def run_benchmark(
    num_games: int = 100,
    num_rows: int = 3,
    num_cols: int = 3,
    num_trials: int = 100,
    payoff_range: tuple = (-100, 100),
    seed: int = None,
    output_dir: str = "results",
    llm_seed: int = None,
    llm_type: str = "dummy",
    llm_model: str = None
):
    """
    Run the benchmark with multiple trials per game.
    
    Args:
        num_games: Number of games to generate
        num_rows: Number of row player actions
        num_cols: Number of column player actions
        num_trials: Number of trials per game
        payoff_range: Range of payoff values
        seed: Seed for game generation
        output_dir: Directory to save results
        llm_seed: Seed for LLM randomness (DummyLLM only)
        llm_type: Type of LLM ('dummy', 'ollama', 'together', 'openai')
        llm_model: Specific model name (optional, uses defaults)
    """
    print(f"Generating {num_games} games ({num_rows}x{num_cols})...")
    games = generate_game_batch(num_games, num_rows, num_cols, payoff_range, seed=seed)
    
    print(f"Initializing {llm_type.upper()} LLM interface...")
    
    # Initialize the appropriate LLM backend
    if llm_type.lower() == "dummy":
        llm = DummyLLM(seed=llm_seed, use_pure_actions=True)
    elif llm_type.lower() == "ollama":
        model = llm_model or "llama3.1"
        print(f"  Model: {model}")
        llm = OllamaLLM(model=model)
    elif llm_type.lower() == "together":
        model = llm_model or "meta-llama/Llama-3-70b-chat-hf"
        print(f"  Model: {model}")
        llm = TogetherAILLM(model=model)
    elif llm_type.lower() == "openai":
        model = llm_model or "gpt-3.5-turbo"
        print(f"  Model: {model}")
        llm = OpenAILLM(model=model)
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")
    
    print("Setting up games and computing Nash equilibria...")
    benchmark = Benchmark(llm)
    games_data = benchmark.setup_games(games, verbose=True)
    
    print(f"\nRunning {num_trials} trials per game...")
    trial_results, summary = benchmark.run_trials(num_trials=num_trials, verbose=True)
    
    # Save results
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Dataset 1: Game mapping (game_id -> payoff matrix, nash equilibrium)
    games_file = Path(output_dir) / f"games_{timestamp}.json"
    games_list = [g.to_dict() for g in games_data]
    with open(games_file, 'w') as f:
        json.dump(games_list, f, indent=2)
    
    # Dataset 2: Trial results (game_id, trial_id, llm_decision, llm_value, br_value, nash_gap)
    trials_file = Path(output_dir) / f"trials_{timestamp}.json"
    trials_list = [r.to_dict() for r in trial_results]
    with open(trials_file, 'w') as f:
        json.dump(trials_list, f, indent=2)
    
    # Summary statistics
    summary_file = Path(output_dir) / f"summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDatasets saved:")
    print(f"  Games: {games_file}")
    print(f"  Trials: {trials_file}")
    print(f"  Summary: {summary_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Number of games: {summary['num_games']}")
    print(f"Trials per game: {summary['num_trials_per_game']}")
    print(f"Total trials: {summary['total_trials']}")
    print(f"Mean Nash gap: {summary['mean_nash_gap']:.4f}")
    print(f"Median Nash gap: {summary['median_nash_gap']:.4f}")
    print(f"Std Nash gap: {summary['std_nash_gap']:.4f}")
    print(f"Min Nash gap: {summary['min_nash_gap']:.4f}")
    print(f"Max Nash gap: {summary['max_nash_gap']:.4f}")
    print(f"Mean LLM value: {summary['mean_llm_value']:.4f}")
    print(f"Mean BR value: {summary['mean_br_value']:.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM game theory benchmark with multiple trials")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games (default: 100)")
    parser.add_argument("--num-trials", type=int, default=100, help="Number of trials per game (default: 100)")
    parser.add_argument("--num-rows", type=int, default=3, help="Number of row actions (default: 3)")
    parser.add_argument("--num-cols", type=int, default=3, help="Number of column actions (default: 3)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for game generation")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory (default: results)")
    parser.add_argument("--llm-seed", type=int, default=None, help="Random seed for LLM")
    parser.add_argument("--llm-type", type=str, default="dummy", 
                       choices=["dummy", "ollama", "together", "openai"],
                       help="LLM backend type (default: dummy)")
    parser.add_argument("--llm-model", type=str, default=None,
                       help="Specific model name (overrides defaults)")
    
    args = parser.parse_args()
    
    run_benchmark(
        num_games=args.num_games,
        num_rows=args.num_rows,
        num_cols=args.num_cols,
        num_trials=args.num_trials,
        seed=args.seed,
        output_dir=args.output_dir,
        llm_seed=args.llm_seed,
        llm_type=args.llm_type,
        llm_model=args.llm_model
    )
