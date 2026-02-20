"""Interface for querying LLMs and parsing responses."""

import numpy as np
import json
from typing import Union, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from src.game_generator import MatrixGame


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def query(self, game: MatrixGame) -> Union[int, np.ndarray]:
        """
        Query the LLM for a decision in the given game.
        
        Args:
            game: MatrixGame instance
            
        Returns:
            Either a pure action (int) or mixed strategy (np.ndarray of probabilities)
        """
        pass


class DummyLLM(LLMInterface):
    """Dummy LLM that plays random strategies for testing."""
    
    def __init__(self, seed: int = None, use_pure_actions: bool = False):
        """
        Initialize dummy LLM.
        
        Args:
            seed: Random seed for reproducibility
            use_pure_actions: If True, only play pure actions; else play mixed strategies
        """
        self.seed = seed
        self.use_pure_actions = use_pure_actions
        if seed is not None:
            np.random.seed(seed)
    
    def query(self, game: MatrixGame) -> Union[int, np.ndarray]:
        """Play a random action."""
        return np.random.randint(0, game.num_row_actions)
    
    def query_mixed_strategy(self, game: MatrixGame) -> np.ndarray:
        """Play a random mixed strategy (Dirichlet distribution)."""
        return np.random.dirichlet(np.ones(game.num_row_actions))


class GamePromptFormatter:
    """Format games into prompts for LLMs."""
    
    @staticmethod
    def format_game_as_text(game: MatrixGame) -> str:
        """
        Format a matrix game as readable text for pure action selection.
        
        Args:
            game: MatrixGame instance
            
        Returns:
            String representation of the game
        """
        lines = ["Zero-sum matrix game:"]
        lines.append("Your payoff matrix (you are the row player):")
        lines.append("")
        
        # Format matrix with action names
        col_indices = [f"  Col{j}" for j in range(game.num_col_actions)]
        lines.append("  " + "".join(f"{idx:>10}" for idx in col_indices))
        
        for i, action_name in enumerate(game.action_names):
            row_values = [f"{game.payoff_matrix[i, j]:>10.2f}" for j in range(game.num_col_actions)]
            lines.append(f"Row{i:>2} " + "".join(row_values))
        
        lines.append("")
        lines.append("Your actions: " + ", ".join(game.action_names))
        lines.append("Opponent has " + str(game.num_col_actions) + " possible actions")
        lines.append("")
        lines.append("IMPORTANT: The opponent is playing their Nash equilibrium mixed strategy.")
        lines.append("What single action do you choose? Respond with just the action number (0, 1, or 2).")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_game_for_mixed_strategy(game: MatrixGame) -> str:
        """
        Format a matrix game as text requesting a mixed strategy (probability distribution).
        
        Args:
            game: MatrixGame instance
            
        Returns:
            String representation requesting mixed strategy in JSON format
        """
        lines = ["Zero-sum matrix game:"]
        lines.append("Your payoff matrix (you are the row player):")
        lines.append("")
        
        # Format matrix with action names
        col_indices = [f"  Col{j}" for j in range(game.num_col_actions)]
        lines.append("  " + "".join(f"{idx:>10}" for idx in col_indices))
        
        for i, action_name in enumerate(game.action_names):
            row_values = [f"{game.payoff_matrix[i, j]:>10.2f}" for j in range(game.num_col_actions)]
            lines.append(f"Row{i:>2} " + "".join(row_values))
        
        lines.append("")
        lines.append("Your actions: " + ", ".join(game.action_names))
        lines.append("Opponent has " + str(game.num_col_actions) + " possible actions")
        lines.append("")
        lines.append("IMPORTANT: The opponent is playing their Nash equilibrium mixed strategy.")
        lines.append("Provide YOUR mixed strategy (probability distribution) as a JSON object.")
        lines.append("The probabilities must sum to 1.0 and be non-negative.")
        lines.append("")
        lines.append('Example response format:')
        lines.append('{"action_0": 0.5, "action_1": 0.3, "action_2": 0.2}')
        lines.append("")
        lines.append("Respond ONLY with valid JSON, no other text.")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_game_as_json(game: MatrixGame) -> str:
        """
        Format a matrix game as JSON.
        
        Args:
            game: MatrixGame instance
            
        Returns:
            JSON string representation of the game
        """
        game_dict = {
            "game_type": "zero_sum_matrix",
            "num_row_actions": int(game.num_row_actions),
            "num_col_actions": int(game.num_col_actions),
            "action_names": game.action_names,
            "payoff_matrix": game.payoff_matrix.tolist()
        }
        return json.dumps(game_dict, indent=2)


class ResponseParser:
    """Parse LLM responses into game decisions."""
    
    @staticmethod
    def parse_response(response: str, num_actions: int) -> Optional[Union[int, np.ndarray]]:
        """
        Parse LLM response to extract pure action.
        
        Args:
            response: LLM response text
            num_actions: Number of possible actions
            
        Returns:
            Pure action (int) or None if parsing fails
        """
        response_lower = response.lower().strip()
        
        # Try to parse as pure action (look for "action X" or numbers)
        for i in range(num_actions):
            if f"action {i}" in response_lower or f"a{i}" in response_lower:
                return i
        
        # Try to extract just a number (0, 1, 2, etc.)
        import re
        match = re.search(r'\b([0-' + str(num_actions-1) + r'])\b', response_lower)
        if match:
            return int(match.group(1))
        
        return None
    
    @staticmethod
    def parse_mixed_strategy(response: str, num_actions: int) -> Optional[np.ndarray]:
        """
        Parse LLM response to extract mixed strategy (probability distribution).
        
        Args:
            response: LLM response text (should be JSON)
            num_actions: Number of possible actions
            
        Returns:
            Mixed strategy as np.ndarray of probabilities, or None if parsing fails
        """
        try:
            # Try to extract JSON from response
            import json as json_lib
            import re
            
            response = response.strip()
            
            # First try: parse as direct JSON
            try:
                data = json_lib.loads(response)
            except:
                # Second try: extract JSON from text
                json_match = re.search(r'\{[^{}]*\}', response)
                if not json_match:
                    return None
                data = json_lib.loads(json_match.group())
            
            # Extract probabilities
            strategy = np.zeros(num_actions)
            
            # Support multiple key formats: "action_0", "action0", "0", etc.
            for i in range(num_actions):
                for key in [f"action_{i}", f"action{i}", f"a{i}", str(i)]:
                    if key in data:
                        strategy[i] = float(data[key])
                        break
            
            # Validate
            if np.any(strategy < 0):
                return None
            
            total = strategy.sum()
            if total <= 0:
                return None
            
            # Normalize to ensure sum = 1
            strategy = strategy / total
            
            return strategy
            
        except Exception as e:
            return None


class TogetherAILLM(LLMInterface):
    """Llama 3 via Together AI (hosted inference)."""
    
    def __init__(self, model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", api_key: str = None):
        """
        Initialize Together AI LLM.
        
        Args:
            model: Model name (Llama 3 variant)
            api_key: Together AI API key (from env variable if not provided)
        """
        import os
        self.model = model
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY env variable not set")
        
        try:
            import together
            self.client = together.Together(api_key=self.api_key)
        except ImportError:
            raise ImportError("together required for TogetherAILLM. Install: pip install together")
    
    def query(self, game: MatrixGame) -> Union[int, np.ndarray]:
        """Query Together AI API for pure action."""
        prompt = GamePromptFormatter.format_game_as_text(game)
        
        try:
            response = self.client.completions.create(
                prompt=prompt,
                model=self.model,
                max_tokens=256,
                temperature=0.7
            )
            # Extract text from response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                text = response.choices[0].text
            else:
                text = str(response)
            
            # Try to parse response
            decision = ResponseParser.parse_response(text, game.num_row_actions)
            if decision is not None:
                return decision
            
            # Fallback: random choice
            return np.random.randint(0, game.num_row_actions)
        except Exception as e:
            print(f"Error querying Together AI: {e}")
            return np.random.randint(0, game.num_row_actions)
    
    def query_mixed_strategy(self, game: MatrixGame) -> np.ndarray:
        """Query Together AI API for mixed strategy (probability distribution)."""
        prompt = GamePromptFormatter.format_game_for_mixed_strategy(game)
        
        try:
            response = self.client.completions.create(
                prompt=prompt,
                model=self.model,
                max_tokens=256,
                temperature=0.7
            )
            # Extract text from response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                text = response.choices[0].text
            else:
                text = str(response)
            
            # Try to parse mixed strategy response
            strategy = ResponseParser.parse_mixed_strategy(text, game.num_row_actions)
            if strategy is not None:
                return strategy
            
            # Fallback: uniform random strategy
            return np.ones(game.num_row_actions) / game.num_row_actions
        except Exception as e:
            print(f"Error querying Together AI for mixed strategy: {e}")
            return np.ones(game.num_row_actions) / game.num_row_actions


class OpenAILLM(LLMInterface):
    """GPT-4 / GPT-3.5 via OpenAI API."""
    
    def __init__(self, model: str = "gpt-4", api_key: str = None):
        """
        Initialize OpenAI LLM.
        
        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (from env variable if not provided)
        """
        import os
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY env variable not set")
        
        try:
            import openai
            openai.api_key = self.api_key
            self.openai = openai
        except ImportError:
            raise ImportError("openai required for OpenAILLM. Install: pip install openai")
    
    def query(self, game: MatrixGame) -> Union[int, np.ndarray]:
        """Query OpenAI API for pure action."""
        prompt = GamePromptFormatter.format_game_as_text(game)
        
        try:
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=256
            )
            text = response["choices"][0]["message"]["content"]
            
            # Try to parse response
            decision = ResponseParser.parse_response(text, game.num_row_actions)
            if decision is not None:
                return decision
            
            # Fallback: random choice
            return np.random.randint(0, game.num_row_actions)
        except Exception as e:
            print(f"Error querying OpenAI: {e}")
            return np.random.randint(0, game.num_row_actions)
    
    def query_mixed_strategy(self, game: MatrixGame) -> np.ndarray:
        """Query OpenAI API for mixed strategy."""
        prompt = GamePromptFormatter.format_game_for_mixed_strategy(game)
        
        try:
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=256
            )
            text = response["choices"][0]["message"]["content"]
            
            # Try to parse mixed strategy response
            strategy = ResponseParser.parse_mixed_strategy(text, game.num_row_actions)
            if strategy is not None:
                return strategy
            
            # Fallback: uniform random strategy
            return np.ones(game.num_row_actions) / game.num_row_actions
        except Exception as e:
            print(f"Error querying OpenAI for mixed strategy: {e}")
            return np.ones(game.num_row_actions) / game.num_row_actions
