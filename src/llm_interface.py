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
        """Play a random strategy."""
        if self.use_pure_actions:
            return np.random.randint(0, game.num_row_actions)
        else:
            strategy = np.random.dirichlet(np.ones(game.num_row_actions))
            return strategy


class GamePromptFormatter:
    """Format games into prompts for LLMs."""
    
    @staticmethod
    def format_game_as_text(game: MatrixGame) -> str:
        """
        Format a matrix game as readable text.
        
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
        lines.append("What action (or mixed strategy) do you choose?")
        
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
        Parse LLM response to extract action or mixed strategy.
        
        Args:
            response: LLM response text
            num_actions: Number of possible actions
            
        Returns:
            Pure action (int) or mixed strategy (np.ndarray), or None if parsing fails
        """
        response_lower = response.lower().strip()
        
        # Try to parse as pure action (look for "action X" or "a0", "a1", etc.)
        for i in range(num_actions):
            if f"action {i}" in response_lower or f"a{i}" in response_lower:
                return i
        
        # Try to parse as mixed strategy from JSON-like format
        try:
            # Look for probability values
            strategy = np.zeros(num_actions)
            found_probabilities = False
            
            for i in range(num_actions):
                for pattern in [f"action {i}:", f"a{i}:", f"action{i}:", f"prob_{i}:"]:
                    if pattern in response_lower:
                        # Try to extract number after pattern
                        idx = response_lower.find(pattern)
                        rest = response_lower[idx + len(pattern):].strip()
                        # Extract first number
                        import re
                        match = re.search(r"[\d.]+", rest)
                        if match:
                            strategy[i] = float(match.group())
                            found_probabilities = True
            
            if found_probabilities:
                strategy = strategy / (strategy.sum() + 1e-10)  # Normalize
                if np.all(strategy >= 0) and np.abs(strategy.sum() - 1.0) < 0.01:
                    return strategy
        except:
            pass
        
        return None


class OllamaLLM(LLMInterface):
    """Llama 3 via Ollama (local inference)."""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama LLM.
        
        Args:
            model: Model name (e.g., "llama2", "llama3", etc.)
            base_url: Ollama API endpoint
        """
        self.model = model
        self.base_url = base_url
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("requests required for OllamaLLM. Install: pip install requests")
    
    def query(self, game: MatrixGame) -> Union[int, np.ndarray]:
        """Query Ollama API."""
        prompt = GamePromptFormatter.format_game_as_text(game)
        
        try:
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=300  # 5 minute timeout for model inference
            )
            response.raise_for_status()
            text = response.json().get("response", "")
            
            # Try to parse response
            decision = ResponseParser.parse_response(text, game.num_row_actions)
            if decision is not None:
                return decision
            
            # Fallback: random choice
            return np.random.randint(0, game.num_row_actions)
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            print(f"Make sure Ollama is running: ollama serve")
            return np.random.randint(0, game.num_row_actions)


class TogetherAILLM(LLMInterface):
    """Llama 3 via Together AI (hosted inference)."""
    
    def __init__(self, model: str = "meta-llama/Llama-3-70b-chat-hf", api_key: str = None):
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
        """Query Together AI API."""
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
        """Query OpenAI API."""
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
