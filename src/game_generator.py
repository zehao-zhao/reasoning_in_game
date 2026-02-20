"""Generate random zero-sum matrix games."""

import numpy as np
from typing import List, Tuple


class MatrixGame:
    """Represents a zero-sum matrix game."""
    
    def __init__(self, payoff_matrix: np.ndarray, action_names: List[str] = None):
        """
        Initialize a matrix game.
        
        Args:
            payoff_matrix: (m, n) array where entry (i, j) is row player's payoff
            action_names: List of action names for row player
        """
        self.payoff_matrix = payoff_matrix
        self.num_row_actions, self.num_col_actions = payoff_matrix.shape
        
        if action_names is None:
            self.action_names = [f"a{i}" for i in range(self.num_row_actions)]
        else:
            self.action_names = action_names
    
    def get_payoff(self, row_action: int, col_action: int) -> float:
        """Get row player's payoff for given action pair."""
        return float(self.payoff_matrix[row_action, col_action])
    
    def __repr__(self):
        return f"MatrixGame({self.num_row_actions}x{self.num_col_actions})\n{self.payoff_matrix}"


def generate_random_game(
    num_rows: int, 
    num_cols: int, 
    payoff_range: Tuple[float, float] = (-100, 100),
    seed: int = None
) -> MatrixGame:
    """
    Generate a random zero-sum matrix game.
    
    Args:
        num_rows: Number of row player actions
        num_cols: Number of column player actions
        payoff_range: (min, max) for payoff values
        seed: Random seed for reproducibility
        
    Returns:
        MatrixGame instance
    """
    if seed is not None:
        np.random.seed(seed)
    
    payoff_matrix = np.random.uniform(
        payoff_range[0], payoff_range[1], size=(num_rows, num_cols)
    )
    
    return MatrixGame(payoff_matrix)


def generate_game_batch(
    num_games: int,
    num_rows: int,
    num_cols: int,
    payoff_range: Tuple[float, float] = (-100, 100),
    seed: int = None
) -> List[MatrixGame]:
    """
    Generate a batch of random zero-sum games.
    
    Args:
        num_games: Number of games to generate
        num_rows: Number of row player actions
        num_cols: Number of column player actions
        payoff_range: (min, max) for payoff values
        seed: Random seed for reproducibility
        
    Returns:
        List of MatrixGame instances
    """
    if seed is not None:
        np.random.seed(seed)
    
    games = []
    for _ in range(num_games):
        games.append(
            generate_random_game(num_rows, num_cols, payoff_range, seed=None)
        )
    
    return games
