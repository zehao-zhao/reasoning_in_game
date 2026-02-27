"""Game bucket definitions and stratified game generation utilities."""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.game_generator import MatrixGame, generate_random_game
from src.nash_solver import NashSolver


@dataclass(frozen=True)
class GameBucket:
    """Configuration for a game family bucket."""

    bucket_id: str
    num_rows: int
    num_cols: int
    payoff_range: Tuple[float, float]
    nash_type: str  # "pure" or "mixed"


@dataclass
class ClassifiedGame:
    """A game and metadata for stratified benchmarking."""

    game_id: int
    bucket_id: str
    seed: int
    payoff_matrix: np.ndarray
    nash_equilibrium_row: np.ndarray
    nash_equilibrium_col: np.ndarray
    equilibrium_type: str
    payoff_std: float

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["payoff_matrix"] = self.payoff_matrix.tolist()
        d["nash_equilibrium_row"] = self.nash_equilibrium_row.tolist()
        d["nash_equilibrium_col"] = self.nash_equilibrium_col.tolist()
        return d


DEFAULT_BUCKETS: List[GameBucket] = [
    GameBucket("2x2_lowVar_pure", 2, 2, (-10, 10), "pure"),
    GameBucket("2x2_lowVar_mixed", 2, 2, (-10, 10), "mixed"),
    GameBucket("3x3_midVar_pure", 3, 3, (-50, 50), "pure"),
    GameBucket("3x3_midVar_mixed", 3, 3, (-50, 50), "mixed"),
    GameBucket("4x4_highVar_pure", 4, 4, (-100, 100), "pure"),
    GameBucket("4x4_highVar_mixed", 4, 4, (-100, 100), "mixed"),
]


def detect_equilibrium_type(strategy: np.ndarray, tol: float = 1e-6) -> str:
    """Classify a mixed strategy as pure or mixed."""
    return "pure" if np.max(strategy) >= 1.0 - tol else "mixed"


def classify_game(matrix_game: MatrixGame, solver: Optional[NashSolver] = None) -> Dict[str, object]:
    """Compute game-level features used for bucketing and analysis."""
    solver = solver or NashSolver()
    nash_row, nash_col = solver.solve_zero_sum_game(matrix_game)
    row_type = detect_equilibrium_type(nash_row)
    col_type = detect_equilibrium_type(nash_col)
    equilibrium_type = "pure" if row_type == "pure" and col_type == "pure" else "mixed"

    return {
        "nash_row": nash_row,
        "nash_col": nash_col,
        "equilibrium_type": equilibrium_type,
        "payoff_std": float(np.std(matrix_game.payoff_matrix)),
    }


def generate_games_for_bucket(
    bucket: GameBucket,
    num_games: int,
    seed: int,
    max_attempts_per_game: int = 400,
) -> List[ClassifiedGame]:
    """Generate games matching bucket constraints."""
    rng = np.random.default_rng(seed)
    solver = NashSolver()
    games: List[ClassifiedGame] = []

    game_idx = 0
    total_attempts = 0
    max_attempts_total = max_attempts_per_game * max(1, num_games)

    while len(games) < num_games and total_attempts < max_attempts_total:
        total_attempts += 1
        sample_seed = int(rng.integers(0, 2**31 - 1))
        game = generate_random_game(
            num_rows=bucket.num_rows,
            num_cols=bucket.num_cols,
            payoff_range=bucket.payoff_range,
            seed=sample_seed,
        )
        info = classify_game(game, solver)

        if info["equilibrium_type"] != bucket.nash_type:
            continue

        games.append(
            ClassifiedGame(
                game_id=game_idx,
                bucket_id=bucket.bucket_id,
                seed=sample_seed,
                payoff_matrix=game.payoff_matrix,
                nash_equilibrium_row=info["nash_row"],
                nash_equilibrium_col=info["nash_col"],
                equilibrium_type=info["equilibrium_type"],
                payoff_std=info["payoff_std"],
            )
        )
        game_idx += 1

    if len(games) < num_games:
        raise RuntimeError(
            f"Could not generate enough games for bucket {bucket.bucket_id}. "
            f"Requested {num_games}, generated {len(games)} after {total_attempts} attempts."
        )

    return games
