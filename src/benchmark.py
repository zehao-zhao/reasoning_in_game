"""Run benchmark and compute Nash gap metrics."""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from src.game_generator import MatrixGame
from src.nash_solver import NashSolver
from src.llm_interface import LLMInterface


@dataclass
class BenchmarkResult:
    """Result for a single game in the benchmark."""
    game_id: int
    llm_decision: Any  # int or np.ndarray
    llm_value: float
    best_response_value: float
    nash_gap: float
    nash_equilibrium_row: np.ndarray
    nash_equilibrium_col: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result_dict = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(result_dict['llm_decision'], np.ndarray):
            result_dict['llm_decision'] = result_dict['llm_decision'].tolist()
        result_dict['nash_equilibrium_row'] = result_dict['nash_equilibrium_row'].tolist()
        result_dict['nash_equilibrium_col'] = result_dict['nash_equilibrium_col'].tolist()
        return result_dict


class Benchmark:
    """Run LLM game theory benchmark."""
    
    def __init__(self, llm: LLMInterface):
        """
        Initialize benchmark.
        
        Args:
            llm: LLMInterface implementation
        """
        self.llm = llm
        self.nash_solver = NashSolver()
    
    def evaluate_game(self, game: MatrixGame, game_id: int = 0) -> BenchmarkResult:
        """
        Evaluate LLM performance on a single game.
        
        Args:
            game: MatrixGame instance
            game_id: Identifier for the game
            
        Returns:
            BenchmarkResult with Nash gap metric
        """
        # Get LLM decision
        llm_decision = self.llm.query(game)
        
        # Compute Nash equilibrium
        nash_row, nash_col = self.nash_solver.solve_zero_sum_game(game)
        
        # Compute LLM value against Nash opponent
        if isinstance(llm_decision, int):
            # Pure action
            llm_strategy = np.zeros(game.num_row_actions)
            llm_strategy[llm_decision] = 1.0
        else:
            # Mixed strategy
            llm_strategy = llm_decision
            llm_strategy = llm_strategy / (llm_strategy.sum() + 1e-10)  # Normalize
        
        llm_value = self.nash_solver.get_game_value(game, llm_strategy, nash_col)
        
        # Compute best response value against Nash opponent
        br_value = self.nash_solver.get_best_response_value(game, nash_col, is_row_player=True)
        
        # Compute Nash gap
        nash_gap = br_value - llm_value
        
        return BenchmarkResult(
            game_id=game_id,
            llm_decision=llm_decision,
            llm_value=float(llm_value),
            best_response_value=float(br_value),
            nash_gap=float(nash_gap),
            nash_equilibrium_row=nash_row,
            nash_equilibrium_col=nash_col
        )
    
    def evaluate_games(self, games: List[MatrixGame], verbose: bool = False) -> Tuple[List[BenchmarkResult], Dict[str, float]]:
        """
        Evaluate LLM performance on a batch of games.
        
        Args:
            games: List of MatrixGame instances
            verbose: If True, print progress
            
        Returns:
            Tuple of (results, summary_stats)
        """
        results = []
        
        for game_id, game in enumerate(games):
            if verbose and game_id % 10 == 0:
                print(f"Evaluating game {game_id}/{len(games)}")
            
            result = self.evaluate_game(game, game_id)
            results.append(result)
        
        # Compute summary statistics
        nash_gaps = np.array([r.nash_gap for r in results])
        llm_values = np.array([r.llm_value for r in results])
        br_values = np.array([r.best_response_value for r in results])
        
        summary = {
            "num_games": len(games),
            "mean_nash_gap": float(np.mean(nash_gaps)),
            "median_nash_gap": float(np.median(nash_gaps)),
            "std_nash_gap": float(np.std(nash_gaps)),
            "min_nash_gap": float(np.min(nash_gaps)),
            "max_nash_gap": float(np.max(nash_gaps)),
            "mean_llm_value": float(np.mean(llm_values)),
            "mean_br_value": float(np.mean(br_values)),
            "mean_gap_ratio": float(np.mean(nash_gaps / (np.abs(br_values) + 1e-10)))
        }
        
        return results, summary
