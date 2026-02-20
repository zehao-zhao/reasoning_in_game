"""Unit tests for game generator and Nash solver."""

import numpy as np
from src.game_generator import MatrixGame, generate_random_game, generate_game_batch
from src.nash_solver import NashSolver
from src.llm_interface import DummyLLM
from src.benchmark import Benchmark


def test_matrix_game_creation():
    """Test creating a matrix game."""
    payoff = np.array([[1, 2], [3, 4]])
    game = MatrixGame(payoff)
    
    assert game.num_row_actions == 2
    assert game.num_col_actions == 2
    assert game.get_payoff(0, 0) == 1
    assert game.get_payoff(1, 1) == 4
    print("✓ test_matrix_game_creation passed")


def test_game_generation():
    """Test generating random games."""
    game = generate_random_game(3, 3, seed=42)
    assert game.num_row_actions == 3
    assert game.num_col_actions == 3
    
    games = generate_game_batch(5, 3, 3, seed=42)
    assert len(games) == 5
    print("✓ test_game_generation passed")


def test_nash_solver():
    """Test Nash equilibrium solver."""
    payoff = np.array([[3, 0], [4, 1]], dtype=float)
    game = MatrixGame(payoff)
    
    row_strat, col_strat = NashSolver.solve_zero_sum_game(game)
    
    # Check that strategies are valid probability distributions
    assert np.allclose(row_strat.sum(), 1.0)
    assert np.allclose(col_strat.sum(), 1.0)
    assert np.all(row_strat >= 0)
    assert np.all(col_strat >= 0)
    print("✓ test_nash_solver passed")


def test_benchmark():
    """Test benchmark evaluation."""
    games = generate_game_batch(5, 2, 2, seed=42)
    llm = DummyLLM(seed=42)
    benchmark = Benchmark(llm)
    
    results, summary = benchmark.evaluate_games(games)
    
    assert len(results) == 5
    assert summary["num_games"] == 5
    assert all(r.nash_gap >= -1e-6 for r in results)  # Nash gap must be non-negative (with small tolerance)
    print("✓ test_benchmark passed")


if __name__ == "__main__":
    test_matrix_game_creation()
    test_game_generation()
    test_nash_solver()
    test_benchmark()
    print("\nAll tests passed! ✓")
