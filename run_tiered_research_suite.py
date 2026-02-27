#!/usr/bin/env python3
"""Tiered, controlled benchmark across game families and model tiers."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.benchmark import Benchmark, GameData
from src.game_catalog import DEFAULT_BUCKETS, ClassifiedGame, GameBucket, generate_games_for_bucket
from src.llm_interface import DummyLLM, OpenAILLM, TogetherAILLM


DEFAULT_TOGETHER_MODEL_TIERS: Dict[str, List[str]] = {
    "A": [
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
    ],
    "B": [
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
    ],
    "C": [
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "Qwen/Qwen2.5-7B-Instruct-Turbo",
    ],
}

# Users should override this with --openai-model-tiers-json for account-accessible models.
DEFAULT_OPENAI_MODEL_TIERS: Dict[str, List[str]] = {
    "A": ["gpt-4o"],
    "B": ["gpt-4o-mini"],
    "C": [],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Tiered game-theory benchmark")
    parser.add_argument("--num-games-per-bucket", type=int, default=30)
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Use one seed for budget-sensitive runs")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--use-dummy", action="store_true", help="Use DummyLLM for local dry-runs")
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["together", "openai"],
        default=["together"],
        help="LLM providers to evaluate (default: together only)",
    )
    parser.add_argument(
        "--bucket-ids",
        nargs="+",
        default=None,
        help="Optional subset of bucket IDs (e.g. 2x2_lowVar_pure 3x3_midVar_mixed)",
    )
    parser.add_argument(
        "--together-model-tiers-json",
        type=str,
        default=None,
        help='JSON string for tier map, e.g. {"A":["model1"],"B":[],"C":[]}',
    )
    parser.add_argument(
        "--openai-model-tiers-json",
        type=str,
        default=None,
        help='JSON string for tier map, e.g. {"A":["gpt-4o"],"B":["gpt-4o-mini"],"C":[]}',
    )
    return parser.parse_args()


def parse_model_tiers(raw_json: str, default_tiers: Dict[str, List[str]]) -> Dict[str, List[str]]:
    if raw_json is None:
        return default_tiers

    parsed = json.loads(raw_json)
    normalized: Dict[str, List[str]] = {"A": [], "B": [], "C": []}
    for tier in normalized:
        values = parsed.get(tier, [])
        if not isinstance(values, list):
            raise ValueError(f"Tier {tier} must be a list")
        normalized[tier] = [str(v) for v in values]
    return normalized


def select_buckets(bucket_ids: List[str] = None) -> List[GameBucket]:
    if not bucket_ids:
        return DEFAULT_BUCKETS

    available = {b.bucket_id: b for b in DEFAULT_BUCKETS}
    missing = [bid for bid in bucket_ids if bid not in available]
    if missing:
        raise ValueError(f"Unknown bucket ids: {missing}. Available: {list(available.keys())}")

    return [available[bid] for bid in bucket_ids]


def build_model_specs(providers: List[str], together_tiers: Dict[str, List[str]], openai_tiers: Dict[str, List[str]]) -> List[Dict[str, str]]:
    specs: List[Dict[str, str]] = []
    for provider in providers:
        tiers = together_tiers if provider == "together" else openai_tiers
        for tier, models in tiers.items():
            for model in models:
                specs.append({"provider": provider, "tier": tier, "model": model})
    return specs


def bucket_to_dict(bucket: GameBucket) -> Dict:
    return {
        "bucket_id": bucket.bucket_id,
        "num_rows": bucket.num_rows,
        "num_cols": bucket.num_cols,
        "payoff_range": list(bucket.payoff_range),
        "nash_type": bucket.nash_type,
    }


def save_games_file(path: Path, games: List[ClassifiedGame]):
    with open(path, "w") as f:
        json.dump([g.to_dict() for g in games], f, indent=2)


def run_model_on_games(
    provider: str,
    model_name: str,
    tier: str,
    games: List[ClassifiedGame],
    num_trials: int,
    num_workers: int,
    temperature: float,
    use_dummy: bool,
):
    if use_dummy:
        llm = DummyLLM(seed=123)
    elif provider == "together":
        llm = TogetherAILLM(model=model_name, temperature=temperature)
    elif provider == "openai":
        llm = OpenAILLM(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    benchmark = Benchmark(llm)
    benchmark.games_data = [
        GameData(
            game_id=g.game_id,
            payoff_matrix=g.payoff_matrix,
            nash_equilibrium_row=g.nash_equilibrium_row,
            nash_equilibrium_col=g.nash_equilibrium_col,
        )
        for g in games
    ]

    pure_trials, pure_summary = benchmark.run_trials_parallel(num_trials=num_trials, num_workers=num_workers)
    mixed_trials, mixed_summary = benchmark.run_trials_parallel_mixed_strategy(num_trials=num_trials, num_workers=num_workers)

    return {
        "provider": provider,
        "tier": tier,
        "model": model_name if not use_dummy else f"dummy_tier_{tier}",
        "pure_summary": pure_summary,
        "mixed_summary": mixed_summary,
        "pure_trials": [t.to_dict() for t in pure_trials],
        "mixed_trials": [t.to_dict() for t in mixed_trials],
    }


def write_csv(path: Path, rows: List[Dict]):
    with open(path, "w", newline="") as f:
        if not rows:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    selected_buckets = select_buckets(args.bucket_ids)
    together_tiers = parse_model_tiers(args.together_model_tiers_json, DEFAULT_TOGETHER_MODEL_TIERS)
    openai_tiers = parse_model_tiers(args.openai_model_tiers_json, DEFAULT_OPENAI_MODEL_TIERS)
    model_specs = build_model_specs(args.providers, together_tiers, openai_tiers)

    if not model_specs:
        raise RuntimeError("No models configured. Provide model tiers via --*_model_tiers_json.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"tiered_suite_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_at": timestamp,
        "num_games_per_bucket": args.num_games_per_bucket,
        "num_trials": args.num_trials,
        "seeds": args.seeds,
        "temperature": args.temperature,
        "num_workers": args.num_workers,
        "providers": args.providers,
        "buckets": [bucket_to_dict(b) for b in selected_buckets],
        "together_model_tiers": together_tiers,
        "openai_model_tiers": openai_tiers,
    }
    with open(run_dir / "suite_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    rows = []

    for seed in args.seeds:
        for bucket in selected_buckets:
            print(f"Generating games for seed={seed}, bucket={bucket.bucket_id}")
            games = generate_games_for_bucket(bucket, num_games=args.num_games_per_bucket, seed=seed)

            games_dir = run_dir / "games" / f"seed_{seed}"
            games_dir.mkdir(parents=True, exist_ok=True)
            save_games_file(games_dir / f"{bucket.bucket_id}.json", games)

            for spec in model_specs:
                print(f"  Running provider={spec['provider']}, tier={spec['tier']}, model={spec['model']}")
                result = run_model_on_games(
                    provider=spec["provider"],
                    model_name=spec["model"],
                    tier=spec["tier"],
                    games=games,
                    num_trials=args.num_trials,
                    num_workers=args.num_workers,
                    temperature=args.temperature,
                    use_dummy=args.use_dummy,
                )

                model_slug = result["model"].replace("/", "__")
                out_dir = run_dir / "model_runs" / f"seed_{seed}" / bucket.bucket_id / result["provider"] / model_slug
                out_dir.mkdir(parents=True, exist_ok=True)

                with open(out_dir / "summary_pure.json", "w") as f:
                    json.dump(result["pure_summary"], f, indent=2)
                with open(out_dir / "summary_mixed.json", "w") as f:
                    json.dump(result["mixed_summary"], f, indent=2)
                with open(out_dir / "trials_pure.json", "w") as f:
                    json.dump(result["pure_trials"], f, indent=2)
                with open(out_dir / "trials_mixed.json", "w") as f:
                    json.dump(result["mixed_trials"], f, indent=2)

                for mode, summary in [("pure", result["pure_summary"]), ("mixed", result["mixed_summary"])]:
                    rows.append(
                        {
                            "seed": seed,
                            "provider": result["provider"],
                            "bucket_id": bucket.bucket_id,
                            "equilibrium_target": bucket.nash_type,
                            "num_rows": bucket.num_rows,
                            "num_cols": bucket.num_cols,
                            "payoff_min": bucket.payoff_range[0],
                            "payoff_max": bucket.payoff_range[1],
                            "tier": result["tier"],
                            "model": result["model"],
                            "mode": mode,
                            "num_games": summary["num_games"],
                            "num_trials_per_game": summary["num_trials_per_game"],
                            "total_trials": summary["total_trials"],
                            "mean_nash_gap": summary["mean_nash_gap"],
                            "median_nash_gap": summary["median_nash_gap"],
                            "std_nash_gap": summary["std_nash_gap"],
                            "min_nash_gap": summary["min_nash_gap"],
                            "max_nash_gap": summary["max_nash_gap"],
                            "mean_llm_value": summary["mean_llm_value"],
                            "reference_value": summary.get("mean_br_value", summary.get("mean_nash_value")),
                            "parse_success_rate": summary.get("parse_success_rate", np.nan),
                            "decision_source_counts": json.dumps(summary.get("decision_source_counts", {}), sort_keys=True),
                        }
                    )

    csv_path = run_dir / "big_table_all_runs.csv"
    write_csv(csv_path, rows)

    agg_keys = [
        "provider",
        "bucket_id",
        "equilibrium_target",
        "num_rows",
        "num_cols",
        "payoff_min",
        "payoff_max",
        "tier",
        "model",
        "mode",
    ]
    grouped: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in agg_keys)
        grouped.setdefault(key, []).append(row)

    agg_rows = []
    for key, group in grouped.items():
        entry = {k: v for k, v in zip(agg_keys, key)}
        for metric in ["mean_nash_gap", "median_nash_gap", "std_nash_gap", "mean_llm_value", "reference_value", "parse_success_rate"]:
            vals = [float(g[metric]) for g in group]
            entry[f"{metric}_mean"] = float(np.mean(vals))
            entry[f"{metric}_std_across_seeds"] = float(np.std(vals))
        entry["seeds"] = ",".join(str(g["seed"]) for g in group)
        agg_rows.append(entry)

    agg_rows.sort(key=lambda r: (r["provider"], r["bucket_id"], r["mode"], r["tier"], r["model"]))
    agg_csv = run_dir / "big_table_aggregated.csv"
    write_csv(agg_csv, agg_rows)

    md_path = run_dir / "big_table_aggregated.md"
    with open(md_path, "w") as f:
        if not agg_rows:
            f.write("No rows generated.\n")
        else:
            headers = [
                "provider",
                "bucket_id",
                "mode",
                "tier",
                "model",
                "mean_nash_gap_mean",
                "mean_nash_gap_std_across_seeds",
                "mean_llm_value_mean",
                "reference_value_mean",
                "parse_success_rate_mean",
            ]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
            for row in agg_rows:
                f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")

    print("\nDone.")
    print(f"Per-run table: {csv_path}")
    print(f"Aggregated table: {agg_csv}")
    print(f"Markdown table: {md_path}")


if __name__ == "__main__":
    main()
