"""
Example: Using different LLM backends with the benchmark.

This shows how to switch between:
1. DummyLLM (random, for testing)
2. TogetherAILLM (hosted Llama 3)
3. OpenAILLM (GPT-4/3.5)
"""

from src.benchmark import Benchmark
from src.game_generator import generate_game_batch
from src.llm_interface import DummyLLM, TogetherAILLM, OpenAILLM


def example_dummy_llm():
    """Example: Using DummyLLM (random strategy)."""
    print("=" * 60)
    print("Example 1: DummyLLM (random baseline)")
    print("=" * 60)
    
    games = generate_game_batch(5, 3, 3, seed=42)
    llm = DummyLLM(seed=42, use_pure_actions=True)
    
    benchmark = Benchmark(llm)
    games_data = benchmark.setup_games(games)
    results, summary = benchmark.run_trials(num_trials=10)
    
    print(f"Mean Nash gap: {summary['mean_nash_gap']:.4f}")
    print()


def example_together_ai():
    """Example: Using TogetherAILLM (hosted Llama)."""
    print("=" * 60)
    print("Example 2: TogetherAILLM (hosted Llama 3)")
    print("=" * 60)
    print("""
Prerequisites:
1. Get API key from https://www.together.ai
2. Set env variable: export TOGETHER_API_KEY="your_key_here"
3. Install client: pip install together

Note: Hosted inference has latency but doesn't require local GPU.
    """)
    
    try:
        games = generate_game_batch(2, 2, 2, seed=42)
        llm = TogetherAILLM(model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
        
        benchmark = Benchmark(llm)
        games_data = benchmark.setup_games(games)
        results, summary = benchmark.run_trials(num_trials=5)
        
        print(f"Mean Nash gap: {summary['mean_nash_gap']:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure TOGETHER_API_KEY is set")
    print()


def example_openai():
    """Example: Using OpenAILLM (GPT-4/3.5)."""
    print("=" * 60)
    print("Example 3: OpenAILLM (GPT-4/GPT-3.5)")
    print("=" * 60)
    print("""
Prerequisites:
1. Get API key from https://platform.openai.com
2. Set env variable: export OPENAI_API_KEY="your_key_here"
3. Install client: pip install openai

Cost: ~$0.02-0.10 per 100 trials (varies by model)
    """)
    
    try:
        games = generate_game_batch(2, 2, 2, seed=42)
        llm = OpenAILLM(model="gpt-3.5-turbo")  # Use GPT-3.5 for lower cost
        
        benchmark = Benchmark(llm)
        games_data = benchmark.setup_games(games)
        results, summary = benchmark.run_trials(num_trials=5)
        
        print(f"Mean Nash gap: {summary['mean_nash_gap']:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure OPENAI_API_KEY is set")
    print()


if __name__ == "__main__":
    print("LLM Backend Examples\n")
    
    # Uncomment whichever example you want to try:
    example_dummy_llm()
    # example_together_ai()
    # example_openai()
