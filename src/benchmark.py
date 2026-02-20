"""Run benchmark and compute Nash gap metrics."""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.game_generator import MatrixGame
from src.nash_solver import NashSolver
from src.llm_interface import LLMInterface


@dataclass
class TrialResult:
    """Result for a single trial of a game."""
    game_id: int
    trial_id: int
    llm_decision: Any  # int or np.ndarray
    llm_value: float
    best_response_value: float
    nash_gap: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result_dict = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(result_dict['llm_decision'], np.ndarray):
            result_dict['llm_decision'] = result_dict['llm_decision'].tolist()
        return result_dict


@dataclass
class GameData:
    """Data structure for storing a single game."""
    game_id: int
    payoff_matrix: np.ndarray
    nash_equilibrium_row: np.ndarray
    nash_equilibrium_col: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'game_id': self.game_id,
            'payoff_matrix': self.payoff_matrix.tolist(),
            'nash_equilibrium_row': self.nash_equilibrium_row.tolist(),
            'nash_equilibrium_col': self.nash_equilibrium_col.tolist()
        }


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
        self.games_data = []  # Store game data with Nash equilibria
    
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
    
    def setup_games(self, games: List[MatrixGame], verbose: bool = False) -> List[GameData]:
        """
        Setup games by computing Nash equilibria and storing game data.
        
        Args:
            games: List of MatrixGame instances
            verbose: If True, print progress
            
        Returns:
            List of GameData instances
        """
        self.games_data = []
        
        for game_id, game in enumerate(games):
            if verbose and game_id % 10 == 0:
                print(f"Computing Nash for game {game_id}/{len(games)}")
            
            # Compute Nash equilibrium
            nash_row, nash_col = self.nash_solver.solve_zero_sum_game(game)
            
            game_data = GameData(
                game_id=game_id,
                payoff_matrix=game.payoff_matrix,
                nash_equilibrium_row=nash_row,
                nash_equilibrium_col=nash_col
            )
            self.games_data.append(game_data)
        
        return self.games_data
    
    def run_trials(self, num_trials: int = 100, verbose: bool = False) -> Tuple[List[TrialResult], Dict[str, float]]:
        """
        Run multiple trials for each game.
        
        Args:
            num_trials: Number of times to query LLM for each game
            verbose: If True, print progress
            
        Returns:
            Tuple of (trial_results, summary_stats)
        """
        if not self.games_data:
            raise RuntimeError("Must call setup_games() first")
        
        trial_results = []
        
        for game_data in self.games_data:
            game = MatrixGame(game_data.payoff_matrix)
            nash_col = game_data.nash_equilibrium_col
            
            if verbose:
                print(f"Running {num_trials} trials for game {game_data.game_id}")
            
            for trial_id in range(num_trials):
                # Get LLM decision
                llm_decision = self.llm.query(game)
                
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
                
                trial_result = TrialResult(
                    game_id=game_data.game_id,
                    trial_id=trial_id,
                    llm_decision=llm_decision,
                    llm_value=float(llm_value),
                    best_response_value=float(br_value),
                    nash_gap=float(nash_gap)
                )
                trial_results.append(trial_result)
        
        # Compute summary statistics
        nash_gaps = np.array([r.nash_gap for r in trial_results])
        llm_values = np.array([r.llm_value for r in trial_results])
        br_values = np.array([r.best_response_value for r in trial_results])
        
        summary = {
            "num_games": len(self.games_data),
            "num_trials_per_game": num_trials,
            "total_trials": len(trial_results),
            "mean_nash_gap": float(np.mean(nash_gaps)),
            "median_nash_gap": float(np.median(nash_gaps)),
            "std_nash_gap": float(np.std(nash_gaps)),
            "min_nash_gap": float(np.min(nash_gaps)),
            "max_nash_gap": float(np.max(nash_gaps)),
            "mean_llm_value": float(np.mean(llm_values)),
            "mean_br_value": float(np.mean(br_values))
        }
        
        return trial_results, summary
    
    def run_trials_parallel(self, num_trials: int = 100, num_workers: int = 4, verbose: bool = False) -> Tuple[List[TrialResult], Dict[str, float]]:
        """
        Run multiple trials for each game using parallel workers (much faster for network-based LLMs).
        
        Args:
            num_trials: Number of times to query LLM for each game
            num_workers: Number of parallel workers (default: 4)
            verbose: If True, print progress
            
        Returns:
            Tuple of (trial_results, summary_stats)
        """
        if not self.games_data:
            raise RuntimeError("Must call setup_games() first")
        
        trial_results = []
        
        def run_single_trial(args):
            """Helper function to run a single trial (for parallel execution)."""
            game_data, trial_id = args
            game = MatrixGame(game_data.payoff_matrix)
            nash_col = game_data.nash_equilibrium_col
            
            # Get LLM decision
            llm_decision = self.llm.query(game)
            
            # Compute LLM value against Nash opponent
            if isinstance(llm_decision, int):
                llm_strategy = np.zeros(game.num_row_actions)
                llm_strategy[llm_decision] = 1.0
            else:
                llm_strategy = llm_decision
                llm_strategy = llm_strategy / (llm_strategy.sum() + 1e-10)
            
            llm_value = self.nash_solver.get_game_value(game, llm_strategy, nash_col)
            br_value = self.nash_solver.get_best_response_value(game, nash_col, is_row_player=True)
            nash_gap = br_value - llm_value
            
            return TrialResult(
                game_id=game_data.game_id,
                trial_id=trial_id,
                llm_decision=llm_decision,
                llm_value=float(llm_value),
                best_response_value=float(br_value),
                nash_gap=float(nash_gap)
            )
        
        # Create list of all (game_data, trial_id) pairs
        all_trials = []
        for game_data in self.games_data:
            for trial_id in range(num_trials):
                all_trials.append((game_data, trial_id))
        
        # Run trials in parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(run_single_trial, trial): trial for trial in all_trials}
            
            for future in as_completed(futures):
                completed += 1
                trial_result = future.result()
                trial_results.append(trial_result)
                
                if verbose and completed % 10 == 0:
                    print(f"  Completed {completed}/{len(all_trials)} trials")
        
        # Sort by game_id and trial_id for consistency
        trial_results.sort(key=lambda r: (r.game_id, r.trial_id))
        
        # Compute summary statistics
        nash_gaps = np.array([r.nash_gap for r in trial_results])
        llm_values = np.array([r.llm_value for r in trial_results])
        br_values = np.array([r.best_response_value for r in trial_results])
        
        summary = {
            "num_games": len(self.games_data),
            "num_trials_per_game": num_trials,
            "total_trials": len(trial_results),
            "mean_nash_gap": float(np.mean(nash_gaps)),
            "median_nash_gap": float(np.median(nash_gaps)),
            "std_nash_gap": float(np.std(nash_gaps)),
            "min_nash_gap": float(np.min(nash_gaps)),
            "max_nash_gap": float(np.max(nash_gaps)),
            "mean_llm_value": float(np.mean(llm_values)),
            "mean_br_value": float(np.mean(br_values))
        }
        
        return trial_results, summary
    
    def run_trials_mixed_strategy(self, num_trials: int = 100, verbose: bool = False) -> Tuple[List[TrialResult], Dict[str, float]]:
        """
        Run multiple trials for mixed strategy queries (LLM outputs probability distribution).
        Gap = Nash_value - LLM_value (how far LLM's mixed strategy is from Nash equilibrium).
        
        Args:
            num_trials: Number of times to query LLM for each game
            verbose: If True, print progress
            
        Returns:
            Tuple of (trial_results, summary_stats)
        """
        if not self.games_data:
            raise RuntimeError("Must call setup_games() first")
        
        trial_results = []
        
        for game_data in self.games_data:
            game = MatrixGame(game_data.payoff_matrix)
            nash_row = game_data.nash_equilibrium_row
            nash_col = game_data.nash_equilibrium_col
            
            if verbose:
                print(f"Running {num_trials} mixed strategy trials for game {game_data.game_id}")
            
            # Compute Nash value (both play Nash)
            nash_value = self.nash_solver.get_game_value(game, nash_row, nash_col)
            
            for trial_id in range(num_trials):
                # Get LLM mixed strategy
                llm_mixed_strategy = self.llm.query_mixed_strategy(game)
                
                # Normalize to ensure sum = 1
                llm_mixed_strategy = llm_mixed_strategy / (llm_mixed_strategy.sum() + 1e-10)
                
                # Compute LLM value when playing mixed strategy against Nash column
                llm_value = self.nash_solver.get_game_value(game, llm_mixed_strategy, nash_col)
                
                # Compute Nash gap: how far LLM is from Nash
                nash_gap = nash_value - llm_value
                
                trial_result = TrialResult(
                    game_id=game_data.game_id,
                    trial_id=trial_id,
                    llm_decision=llm_mixed_strategy,  # Store the mixed strategy
                    llm_value=float(llm_value),
                    best_response_value=float(nash_value),  # Store Nash value as "best response" for comparison
                    nash_gap=float(nash_gap)
                )
                trial_results.append(trial_result)
        
        # Compute summary statistics
        nash_gaps = np.array([r.nash_gap for r in trial_results])
        llm_values = np.array([r.llm_value for r in trial_results])
        nash_values = np.array([r.best_response_value for r in trial_results])
        
        summary = {
            "num_games": len(self.games_data),
            "num_trials_per_game": num_trials,
            "total_trials": len(trial_results),
            "mode": "mixed_strategy",
            "mean_nash_gap": float(np.mean(nash_gaps)),
            "median_nash_gap": float(np.median(nash_gaps)),
            "std_nash_gap": float(np.std(nash_gaps)),
            "min_nash_gap": float(np.min(nash_gaps)),
            "max_nash_gap": float(np.max(nash_gaps)),
            "mean_llm_value": float(np.mean(llm_values)),
            "mean_nash_value": float(np.mean(nash_values))
        }
        
        return trial_results, summary
    
    def run_trials_parallel_mixed_strategy(self, num_trials: int = 100, num_workers: int = 4, verbose: bool = False) -> Tuple[List[TrialResult], Dict[str, float]]:
        """
        Run multiple mixed strategy trials in parallel (faster for network-based LLMs).
        Gap = Nash_value - LLM_value.
        
        Args:
            num_trials: Number of times to query LLM for each game
            num_workers: Number of parallel workers
            verbose: If True, print progress
            
        Returns:
            Tuple of (trial_results, summary_stats)
        """
        if not self.games_data:
            raise RuntimeError("Must call setup_games() first")
        
        trial_results = []
        
        def run_single_mixed_trial(args):
            """Helper function to run a single mixed strategy trial."""
            game_data, trial_id = args
            game = MatrixGame(game_data.payoff_matrix)
            nash_row = game_data.nash_equilibrium_row
            nash_col = game_data.nash_equilibrium_col
            
            # Compute Nash value
            nash_value = self.nash_solver.get_game_value(game, nash_row, nash_col)
            
            # Get LLM mixed strategy
            llm_mixed_strategy = self.llm.query_mixed_strategy(game)
            llm_mixed_strategy = llm_mixed_strategy / (llm_mixed_strategy.sum() + 1e-10)
            
            # Compute LLM value
            llm_value = self.nash_solver.get_game_value(game, llm_mixed_strategy, nash_col)
            
            # Compute gap
            nash_gap = nash_value - llm_value
            
            return TrialResult(
                game_id=game_data.game_id,
                trial_id=trial_id,
                llm_decision=llm_mixed_strategy,
                llm_value=float(llm_value),
                best_response_value=float(nash_value),
                nash_gap=float(nash_gap)
            )
        
        # Create list of all (game_data, trial_id) pairs
        all_trials = []
        for game_data in self.games_data:
            for trial_id in range(num_trials):
                all_trials.append((game_data, trial_id))
        
        # Run trials in parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(run_single_mixed_trial, trial): trial for trial in all_trials}
            
            for future in as_completed(futures):
                completed += 1
                trial_result = future.result()
                trial_results.append(trial_result)
                
                if verbose and completed % 10 == 0:
                    print(f"  Completed {completed}/{len(all_trials)} trials")
        
        # Sort by game_id and trial_id
        trial_results.sort(key=lambda r: (r.game_id, r.trial_id))
        
        # Compute summary statistics
        nash_gaps = np.array([r.nash_gap for r in trial_results])
        llm_values = np.array([r.llm_value for r in trial_results])
        nash_values = np.array([r.best_response_value for r in trial_results])
        
        summary = {
            "num_games": len(self.games_data),
            "num_trials_per_game": num_trials,
            "total_trials": len(trial_results),
            "mode": "mixed_strategy",
            "mean_nash_gap": float(np.mean(nash_gaps)),
            "median_nash_gap": float(np.median(nash_gaps)),
            "std_nash_gap": float(np.std(nash_gaps)),
            "min_nash_gap": float(np.min(nash_gaps)),
            "max_nash_gap": float(np.max(nash_gaps)),
            "mean_llm_value": float(np.mean(llm_values)),
            "mean_nash_value": float(np.mean(nash_values))
        }
        
        return trial_results, summary
