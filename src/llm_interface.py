"""Interface for querying LLMs and parsing responses."""

from abc import ABC, abstractmethod
import json
from typing import Any, Dict, Optional, Union

import numpy as np

from src.game_generator import MatrixGame


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""

    def __init__(self):
        self._last_query_metadata = {"source": "unknown", "error": None}

    def get_last_query_metadata(self) -> Dict[str, Any]:
        """Metadata about the most recent query call."""
        return dict(self._last_query_metadata)

    @abstractmethod
    def query(self, game: MatrixGame) -> Union[int, np.ndarray]:
        """Query the LLM for a pure action decision."""


class DummyLLM(LLMInterface):
    """Dummy LLM that plays random strategies for testing."""

    def __init__(self, seed: int = None, use_pure_actions: bool = False):
        super().__init__()
        self.seed = seed
        self.use_pure_actions = use_pure_actions
        if seed is not None:
            np.random.seed(seed)

    def query(self, game: MatrixGame) -> Union[int, np.ndarray]:
        self._last_query_metadata = {"source": "dummy_random", "error": None}
        return np.random.randint(0, game.num_row_actions)

    def query_mixed_strategy(self, game: MatrixGame) -> np.ndarray:
        self._last_query_metadata = {"source": "dummy_random", "error": None}
        return np.random.dirichlet(np.ones(game.num_row_actions))


class GamePromptFormatter:
    """Format games into prompts for LLMs."""

    @staticmethod
    def format_game_as_text(game: MatrixGame) -> str:
        lines = ["Zero-sum matrix game:"]
        lines.append("Your payoff matrix (you are the row player):")
        lines.append("")

        col_indices = [f"  Col{j}" for j in range(game.num_col_actions)]
        lines.append("  " + "".join(f"{idx:>10}" for idx in col_indices))

        for i in range(game.num_row_actions):
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
        lines = ["Zero-sum matrix game:"]
        lines.append("Your payoff matrix (you are the row player):")
        lines.append("")

        col_indices = [f"  Col{j}" for j in range(game.num_col_actions)]
        lines.append("  " + "".join(f"{idx:>10}" for idx in col_indices))

        for i in range(game.num_row_actions):
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
        lines.append("Example response format:")
        lines.append('{"action_0": 0.5, "action_1": 0.3, "action_2": 0.2}')
        lines.append("")
        lines.append("Respond ONLY with valid JSON, no other text.")

        return "\n".join(lines)

    @staticmethod
    def format_game_as_json(game: MatrixGame) -> str:
        game_dict = {
            "game_type": "zero_sum_matrix",
            "num_row_actions": int(game.num_row_actions),
            "num_col_actions": int(game.num_col_actions),
            "action_names": game.action_names,
            "payoff_matrix": game.payoff_matrix.tolist(),
        }
        return json.dumps(game_dict, indent=2)


class ResponseParser:
    """Parse LLM responses into game decisions."""

    @staticmethod
    def parse_response(response: str, num_actions: int) -> Optional[Union[int, np.ndarray]]:
        response_lower = response.lower().strip()

        for i in range(num_actions):
            if f"action {i}" in response_lower or f"a{i}" in response_lower:
                return i

        import re

        match = re.search(r"\b([0-" + str(num_actions - 1) + r"])\b", response_lower)
        if match:
            return int(match.group(1))

        return None

    @staticmethod
    def parse_mixed_strategy(response: str, num_actions: int) -> Optional[np.ndarray]:
        try:
            import json as json_lib
            import re

            response = response.strip()

            try:
                data = json_lib.loads(response)
            except Exception:
                json_match = re.search(r"\{[^{}]*\}", response)
                if not json_match:
                    return None
                data = json_lib.loads(json_match.group())

            strategy = np.zeros(num_actions)
            for i in range(num_actions):
                for key in [f"action_{i}", f"action{i}", f"a{i}", str(i)]:
                    if key in data:
                        strategy[i] = float(data[key])
                        break

            if np.any(strategy < 0):
                return None

            total = strategy.sum()
            if total <= 0:
                return None

            return strategy / total
        except Exception:
            return None


class TogetherAILLM(LLMInterface):
    """Llama 3 via Together AI (hosted inference)."""

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        api_key: str = None,
        temperature: float = 0.0,
    ):
        super().__init__()
        import os

        self.model = model
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.temperature = temperature

        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY env variable not set")

        try:
            import together

            self.client = together.Together(api_key=self.api_key)
        except ImportError:
            raise ImportError("together required for TogetherAILLM. Install: pip install together")

    def query(self, game: MatrixGame) -> Union[int, np.ndarray]:
        prompt = GamePromptFormatter.format_game_as_text(game)
        try:
            response = self.client.completions.create(
                prompt=prompt,
                model=self.model,
                max_tokens=256,
                temperature=self.temperature,
            )
            text = response.choices[0].text if hasattr(response, "choices") and response.choices else str(response)

            decision = ResponseParser.parse_response(text, game.num_row_actions)
            if decision is not None:
                self._last_query_metadata = {"source": "parsed", "error": None}
                return decision

            self._last_query_metadata = {"source": "fallback_random", "error": None}
            return np.random.randint(0, game.num_row_actions)
        except Exception as e:
            print(f"Error querying Together AI: {e}")
            self._last_query_metadata = {"source": "error_random", "error": str(e)}
            return np.random.randint(0, game.num_row_actions)

    def query_mixed_strategy(self, game: MatrixGame) -> np.ndarray:
        prompt = GamePromptFormatter.format_game_for_mixed_strategy(game)
        try:
            response = self.client.completions.create(
                prompt=prompt,
                model=self.model,
                max_tokens=256,
                temperature=self.temperature,
            )
            text = response.choices[0].text if hasattr(response, "choices") and response.choices else str(response)

            strategy = ResponseParser.parse_mixed_strategy(text, game.num_row_actions)
            if strategy is not None:
                self._last_query_metadata = {"source": "parsed", "error": None}
                return strategy

            self._last_query_metadata = {"source": "fallback_uniform", "error": None}
            return np.ones(game.num_row_actions) / game.num_row_actions
        except Exception as e:
            print(f"Error querying Together AI for mixed strategy: {e}")
            self._last_query_metadata = {"source": "error_uniform", "error": str(e)}
            return np.ones(game.num_row_actions) / game.num_row_actions


class OpenAILLM(LLMInterface):
    """GPT-4 / GPT-3.5 via OpenAI API."""

    def __init__(self, model: str = "gpt-4", api_key: str = None, temperature: float = 0.0):
        super().__init__()
        import os

        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY env variable not set")

        try:
            import openai

            openai.api_key = self.api_key
            self.openai = openai
        except ImportError:
            raise ImportError("openai required for OpenAILLM. Install: pip install openai")

    def query(self, game: MatrixGame) -> Union[int, np.ndarray]:
        prompt = GamePromptFormatter.format_game_as_text(game)
        try:
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=256,
            )
            text = response["choices"][0]["message"]["content"]

            decision = ResponseParser.parse_response(text, game.num_row_actions)
            if decision is not None:
                self._last_query_metadata = {"source": "parsed", "error": None}
                return decision

            self._last_query_metadata = {"source": "fallback_random", "error": None}
            return np.random.randint(0, game.num_row_actions)
        except Exception as e:
            print(f"Error querying OpenAI: {e}")
            self._last_query_metadata = {"source": "error_random", "error": str(e)}
            return np.random.randint(0, game.num_row_actions)

    def query_mixed_strategy(self, game: MatrixGame) -> np.ndarray:
        prompt = GamePromptFormatter.format_game_for_mixed_strategy(game)
        try:
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=256,
            )
            text = response["choices"][0]["message"]["content"]

            strategy = ResponseParser.parse_mixed_strategy(text, game.num_row_actions)
            if strategy is not None:
                self._last_query_metadata = {"source": "parsed", "error": None}
                return strategy

            self._last_query_metadata = {"source": "fallback_uniform", "error": None}
            return np.ones(game.num_row_actions) / game.num_row_actions
        except Exception as e:
            print(f"Error querying OpenAI for mixed strategy: {e}")
            self._last_query_metadata = {"source": "error_uniform", "error": str(e)}
            return np.ones(game.num_row_actions) / game.num_row_actions
