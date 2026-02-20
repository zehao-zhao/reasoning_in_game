"""Compute Nash equilibrium for zero-sum games."""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional
from src.game_generator import MatrixGame


class NashSolver:
    """Solver for Nash equilibrium in zero-sum games."""
    
    @staticmethod
    def solve_zero_sum_game(game: MatrixGame, epsilon: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mixed strategy Nash equilibrium for zero-sum game.
        
        Uses linear programming to find the Nash equilibrium:
        - For row player: maximize min expected payoff
        - For column player: minimize max expected loss
        
        Args:
            game: MatrixGame instance
            epsilon: Tolerance for numerical stability
            
        Returns:
            Tuple of (row_strategy, col_strategy) as probability distributions
        """
        U = game.payoff_matrix
        m, n = U.shape  # m = num_rows, n = num_cols
        
        # Solve for row player's maxmin strategy
        # max_π1 min_b π1^T U e_b
        # Reformulate as LP: max v such that U^T π1 >= v * 1
        c = np.zeros(m + 1)
        c[-1] = -1  # Maximize v means minimize -v
        
        # Constraints: -U^T π1 + v * 1 <= 0, sum(π1) = 1, π1 >= 0
        A_ub = np.column_stack([-U.T, np.ones(n)])
        b_ub = np.zeros(n)
        
        A_eq = np.ones((1, m + 1))
        A_eq[0, -1] = 0
        b_eq = np.array([1.0])
        
        bounds = [(0, None) for _ in range(m)] + [(None, None)]
        
        result_row = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
        
        if not result_row.success:
            raise RuntimeError("Failed to solve for row player's strategy")
        
        row_strategy = result_row.x[:m]
        row_strategy = np.maximum(row_strategy, 0)  # Ensure non-negative
        row_strategy /= row_strategy.sum() + epsilon
        
        # Solve for column player's minmax strategy
        # min_π2 max_a π2^T U^T e_a = min_π2 max_a e_a^T U π2
        # Reformulate as LP: min u such that U π2 <= u * 1
        c = np.zeros(n + 1)
        c[-1] = 1  # Minimize u
        
        # Constraints: U π2 - u * 1 <= 0, sum(π2) = 1, π2 >= 0
        A_ub = np.column_stack([U, -np.ones(m)])
        b_ub = np.zeros(m)
        
        A_eq = np.ones((1, n + 1))
        A_eq[0, -1] = 0
        b_eq = np.array([1.0])
        
        bounds = [(0, None) for _ in range(n)] + [(None, None)]
        
        result_col = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
        
        if not result_col.success:
            raise RuntimeError("Failed to solve for column player's strategy")
        
        col_strategy = result_col.x[:n]
        col_strategy = np.maximum(col_strategy, 0)  # Ensure non-negative
        col_strategy /= col_strategy.sum() + epsilon
        
        return row_strategy, col_strategy
    
    @staticmethod
    def get_game_value(game: MatrixGame, row_strategy: np.ndarray, col_strategy: np.ndarray) -> float:
        """
        Compute the expected payoff for row player given strategies.
        
        Args:
            game: MatrixGame instance
            row_strategy: Row player's mixed strategy
            col_strategy: Column player's mixed strategy
            
        Returns:
            Expected payoff for row player
        """
        return float(row_strategy @ game.payoff_matrix @ col_strategy)
    
    @staticmethod
    def get_best_response(game: MatrixGame, opponent_strategy: np.ndarray, is_row_player: bool = True) -> int:
        """
        Compute best response pure action against opponent's mixed strategy.
        
        Args:
            game: MatrixGame instance
            opponent_strategy: Opponent's mixed strategy
            is_row_player: If True, compute row player's BR; else column player's BR
            
        Returns:
            Index of best response action
        """
        if is_row_player:
            # Row player's payoff against col_strategy
            payoffs = game.payoff_matrix @ opponent_strategy
        else:
            # Column player's payoff (negative of row payoff) against row_strategy
            payoffs = -(game.payoff_matrix.T @ opponent_strategy)
        
        return int(np.argmax(payoffs))
    
    @staticmethod
    def get_best_response_value(game: MatrixGame, opponent_strategy: np.ndarray, is_row_player: bool = True) -> float:
        """
        Compute best response value against opponent's mixed strategy.
        
        Args:
            game: MatrixGame instance
            opponent_strategy: Opponent's mixed strategy
            is_row_player: If True, compute row player's BR value; else column player's BR value
            
        Returns:
            Best response payoff
        """
        br_action = NashSolver.get_best_response(game, opponent_strategy, is_row_player)
        
        if is_row_player:
            return float(game.payoff_matrix[br_action] @ opponent_strategy)
        else:
            return float(-game.payoff_matrix[:, br_action] @ opponent_strategy)
