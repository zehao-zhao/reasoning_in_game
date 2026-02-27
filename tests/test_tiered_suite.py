"""Tests for tiered benchmark utilities."""

from run_tiered_research_suite import build_model_specs, parse_model_tiers, select_buckets
from src.game_catalog import GameBucket, classify_game, generate_games_for_bucket
from src.game_generator import generate_random_game


def test_classify_game_outputs_required_fields():
    game = generate_random_game(3, 3, seed=123)
    info = classify_game(game)

    assert "nash_row" in info
    assert "nash_col" in info
    assert info["equilibrium_type"] in {"pure", "mixed"}
    assert isinstance(info["payoff_std"], float)


def test_generate_games_for_bucket_respects_bucket_type():
    bucket = GameBucket("small_mixed", 2, 2, (-10, 10), "mixed")
    games = generate_games_for_bucket(bucket, num_games=2, seed=42, max_attempts_per_game=2000)

    assert len(games) == 2
    for g in games:
        assert g.bucket_id == "small_mixed"
        assert g.equilibrium_type == "mixed"


def test_bucket_selection_subset():
    buckets = select_buckets(["2x2_lowVar_pure", "3x3_midVar_mixed"])
    assert [b.bucket_id for b in buckets] == ["2x2_lowVar_pure", "3x3_midVar_mixed"]


def test_model_tier_parsing_and_spec_building():
    together = parse_model_tiers('{"A":["m1"],"B":[],"C":["m2"]}', {"A": [], "B": [], "C": []})
    openai = parse_model_tiers('{"A":["o1"],"B":["o2"],"C":[]}', {"A": [], "B": [], "C": []})
    specs = build_model_specs(["together", "openai"], together, openai)
    assert {s["model"] for s in specs} == {"m1", "m2", "o1", "o2"}
    assert {s["provider"] for s in specs} == {"together", "openai"}
