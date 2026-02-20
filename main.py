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
    llm_model: str = None,
    parallel: bool = False,
    num_workers: int = 14,
    mixed_strategy: bool = False
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
        parallel: If True, use parallel workers for faster LLM queries
        num_workers: Number of parallel workers (default: 14)
        mixed_strategy: If True, LLM outputs mixed strategy (probabilities); else pure actions
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
        model = llm_model or "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
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
    
    # Auto-enable parallelization for network-based LLMs (they have network latency)
    use_parallel = parallel or llm_type.lower() in ["together", "openai"]
    if use_parallel and not parallel and llm_type.lower() in ["together", "openai"]:
        print(f"Auto-enabling parallel execution for {llm_type} (network-based LLM)")
    
    strategy_type = "Mixed Strategy" if mixed_strategy else "Pure Actions"
    print(f"\nRunning {num_trials} trials per game ({strategy_type}){'(parallel with ' + str(num_workers) + ' workers)...' if use_parallel else '...'}")
    
    if mixed_strategy:
        # Query LLM for mixed strategies
        if use_parallel:
            trial_results, summary = benchmark.run_trials_parallel_mixed_strategy(num_trials=num_trials, num_workers=num_workers, verbose=True)
        else:
            trial_results, summary = benchmark.run_trials_mixed_strategy(num_trials=num_trials, verbose=True)
    else:
        # Query LLM for pure actions
        if use_parallel:
            trial_results, summary = benchmark.run_trials_parallel(num_trials=num_trials, num_workers=num_workers, verbose=True)
        else:
            trial_results, summary = benchmark.run_trials(num_trials=num_trials, verbose=True)
    
    # Save results in run-specific folder structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset 1: Game mapping (game_id -> payoff matrix, nash equilibrium)
    games_file = run_dir / "games.json"
    games_list = [g.to_dict() for g in games_data]
    with open(games_file, 'w') as f:
        json.dump(games_list, f, indent=2)
    
    # Dataset 2: Trial results (game_id, trial_id, llm_decision, llm_value, br_value, nash_gap)
    trials_file = run_dir / "trials.json"
    trials_list = [r.to_dict() for r in trial_results]
    with open(trials_file, 'w') as f:
        json.dump(trials_list, f, indent=2)
    
    # Summary statistics
    summary_file = run_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDatasets saved in: {run_dir}/")
    print(f"  ✓ games.json")
    print(f"  ✓ trials.json")
    print(f"  ✓ summary.json")
    
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
    
    if mixed_strategy:
        print(f"Mean Nash value: {summary.get('mean_nash_value', summary.get('mean_br_value', 'N/A')):.4f}")
    else:
        print(f"Mean BR value: {summary['mean_br_value']:.4f}")
    print("="*60)


def run_combined_benchmark(
    num_games: int = 100,
    num_rows: int = 3,
    num_cols: int = 3,
    num_trials: int = 100,
    payoff_range: tuple = (-100, 100),
    seed: int = None,
    output_dir: str = "results",
    llm_seed: int = None,
    llm_type: str = "dummy",
    llm_model: str = None,
    parallel: bool = False,
    num_workers: int = 14
):
    """
    Run BOTH pure action AND mixed strategy benchmarks on the same games.
    
    Args:
        num_games: Number of games to generate
        num_rows: Number of row player actions
        num_cols: Number of column player actions
        num_trials: Number of trials per game
        payoff_range: Range of payoff values
        seed: Seed for game generation
        output_dir: Directory to save results
        llm_seed: Seed for LLM randomness
        llm_type: Type of LLM ('dummy', 'ollama', 'together', 'openai')
        llm_model: Specific model name (optional, uses defaults)
        parallel: If True, use parallel workers
        num_workers: Number of parallel workers (default: 14)
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
        model = llm_model or "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
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
    
    # Auto-enable parallelization for network-based LLMs
    use_parallel = parallel or llm_type.lower() in ["together", "openai"]
    if use_parallel and not parallel and llm_type.lower() in ["together", "openai"]:
        print(f"Auto-enabling parallel execution for {llm_type} (network-based LLM)")
    
    # Create combined output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_dir = Path(output_dir) / f"pure_and_mixed_{timestamp}"
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    # Save shared games file
    games_file = combined_dir / "games.json"
    games_list = [g.to_dict() for g in games_data]
    with open(games_file, 'w') as f:
        json.dump(games_list, f, indent=2)
    print(f"\n✓ Generated {num_games} games")
    print(f"✓ Saved to: {combined_dir}/games.json")
    
    # ========== PURE ACTIONS BENCHMARK ==========
    print(f"\n{'='*60}")
    print("Running PURE ACTIONS benchmark (LLM selects single action)")
    print(f"{'='*60}")
    
    if use_parallel:
        pure_trials, pure_summary = benchmark.run_trials_parallel(num_trials=num_trials, num_workers=num_workers, verbose=True)
    else:
        pure_trials, pure_summary = benchmark.run_trials(num_trials=num_trials, verbose=True)
    
    # Save pure action results
    pure_trials_file = combined_dir / "trials_pure_actions.json"
    pure_trials_list = [r.to_dict() for r in pure_trials]
    with open(pure_trials_file, 'w') as f:
        json.dump(pure_trials_list, f, indent=2)
    
    pure_summary_file = combined_dir / "summary_pure_actions.json"
    with open(pure_summary_file, 'w') as f:
        json.dump(pure_summary, f, indent=2)
    
    print(f"\n✓ Pure action results saved")
    
    # Print pure action summary
    print("\n" + "="*60)
    print("PURE ACTIONS SUMMARY")
    print("="*60)
    print(f"Number of games: {pure_summary['num_games']}")
    print(f"Trials per game: {pure_summary['num_trials_per_game']}")
    print(f"Total trials: {pure_summary['total_trials']}")
    print(f"Mean Nash gap: {pure_summary['mean_nash_gap']:.4f}")
    print(f"Median Nash gap: {pure_summary['median_nash_gap']:.4f}")
    print(f"Mean LLM value: {pure_summary['mean_llm_value']:.4f}")
    print(f"Mean BR value: {pure_summary['mean_br_value']:.4f}")
    print("="*60)
    
    # ========== MIXED STRATEGY BENCHMARK ==========
    print(f"\n{'='*60}")
    print("Running MIXED STRATEGY benchmark (LLM outputs probabilities)")
    print(f"{'='*60}")
    
    if use_parallel:
        mixed_trials, mixed_summary = benchmark.run_trials_parallel_mixed_strategy(num_trials=num_trials, num_workers=num_workers, verbose=True)
    else:
        mixed_trials, mixed_summary = benchmark.run_trials_mixed_strategy(num_trials=num_trials, verbose=True)
    
    # Save mixed strategy results
    mixed_trials_file = combined_dir / "trials_mixed_strategy.json"
    mixed_trials_list = [r.to_dict() for r in mixed_trials]
    with open(mixed_trials_file, 'w') as f:
        json.dump(mixed_trials_list, f, indent=2)
    
    mixed_summary_file = combined_dir / "summary_mixed_strategy.json"
    with open(mixed_summary_file, 'w') as f:
        json.dump(mixed_summary, f, indent=2)
    
    print(f"\n✓ Mixed strategy results saved")
    
    # Print mixed strategy summary
    print("\n" + "="*60)
    print("MIXED STRATEGY SUMMARY")
    print("="*60)
    print(f"Number of games: {mixed_summary['num_games']}")
    print(f"Trials per game: {mixed_summary['num_trials_per_game']}")
    print(f"Total trials: {mixed_summary['total_trials']}")
    print(f"Mean Nash gap: {mixed_summary['mean_nash_gap']:.4f}")
    print(f"Median Nash gap: {mixed_summary['median_nash_gap']:.4f}")
    print(f"Mean LLM value: {mixed_summary['mean_llm_value']:.4f}")
    print(f"Mean Nash value: {mixed_summary.get('mean_nash_value', 'N/A'):.4f}")
    print("="*60)
    
    # ========== COMPARISON ==========
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS: Pure Actions vs Mixed Strategies")
    print(f"{'='*60}")
    
    pure_gap = pure_summary['mean_nash_gap']
    mixed_gap = mixed_summary['mean_nash_gap']
    gap_reduction = (pure_gap - mixed_gap) / pure_gap * 100 if pure_gap > 0 else 0
    
    pure_value = pure_summary['mean_llm_value']
    mixed_value = mixed_summary['mean_llm_value']
    value_improvement = (mixed_value - pure_value) / abs(pure_value) * 100 if pure_value != 0 else 0
    
    print(f"Mean Nash Gap:")
    print(f"  Pure Actions:    {pure_gap:.4f}")
    print(f"  Mixed Strategy:  {mixed_gap:.4f}")
    print(f"  Improvement:     {gap_reduction:.1f}% better")
    print()
    print(f"Mean LLM Value (higher is better):")
    print(f"  Pure Actions:    {pure_value:.4f}")
    print(f"  Mixed Strategy:  {mixed_value:.4f}")
    print(f"  Improvement:     {value_improvement:+.1f}%")
    print("="*60)
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Combined dataset saved in: {combined_dir}/")
    print(f"  ✓ games.json")
    print(f"  ✓ trials_pure_actions.json")
    print(f"  ✓ summary_pure_actions.json")
    print(f"  ✓ trials_mixed_strategy.json")
    print(f"  ✓ summary_mixed_strategy.json")
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
    parser.add_argument("--parallel", action="store_true",
                       help="Use parallel workers for faster LLM queries (useful for network-based LLMs)")
    parser.add_argument("--num-workers", type=int, default=14,
                       help="Number of parallel workers (default: 14 - matches your CPU cores)")
    parser.add_argument("--mixed-strategy", action="store_true",
                       help="Query LLM for mixed strategy (probability distribution) instead of pure actions")
    parser.add_argument("--combined", action="store_true",
                       help="Run BOTH pure actions AND mixed strategy benchmarks on same games (ignore --mixed-strategy)")
    
    args = parser.parse_args()
    
    if args.combined:
        # Run combined pure + mixed benchmark
        run_combined_benchmark(
            num_games=args.num_games,
            num_rows=args.num_rows,
            num_cols=args.num_cols,
            num_trials=args.num_trials,
            seed=args.seed,
            output_dir=args.output_dir,
            llm_seed=args.llm_seed,
            llm_type=args.llm_type,
            llm_model=args.llm_model,
            parallel=args.parallel,
            num_workers=args.num_workers
        )
    else:
        # Run single benchmark (pure or mixed based on flag)
        run_benchmark(
            num_games=args.num_games,
            num_rows=args.num_rows,
            num_cols=args.num_cols,
            num_trials=args.num_trials,
            seed=args.seed,
            output_dir=args.output_dir,
            llm_seed=args.llm_seed,
            llm_type=args.llm_type,
            llm_model=args.llm_model,
            parallel=args.parallel,
            num_workers=args.num_workers,
            mixed_strategy=args.mixed_strategy
        )
