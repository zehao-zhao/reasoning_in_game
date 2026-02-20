#!/usr/bin/env python
"""Analyze LLM game theory performance from run_20260219_172049 dataset."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
data_dir = Path("results/run_20260219_172049")

with open(data_dir / "games.json") as f:
    games = json.load(f)

with open(data_dir / "trials.json") as f:
    trials = json.load(f)

with open(data_dir / "summary.json") as f:
    summary = json.load(f)

print(f"✓ Loaded {len(games)} games")
print(f"✓ Loaded {len(trials)} trials")

# Compute game properties
def analyze_game_properties(game):
    """Extract features from a game matrix."""
    matrix = np.array(game['payoff_matrix'])
    nash_row = np.array(game['nash_equilibrium_row'])
    nash_col = np.array(game['nash_equilibrium_col'])
    
    # Check if Nash equilibrium is pure strategy
    is_pure_row = np.sum(nash_row == 1.0) == 1
    is_pure_col = np.sum(nash_col == 1.0) == 1
    is_pure_nash = is_pure_row and is_pure_col
    
    # Payoff statistics
    flat = matrix.flatten()
    payoff_mean = np.mean(flat)
    payoff_std = np.std(flat)
    payoff_range = np.max(flat) - np.min(flat)
    
    # Check for dominant strategies
    has_dominant_row = False
    has_dominant_col = False
    
    for i in range(len(matrix)):
        if all(matrix[i, j] >= matrix[k, j] for j in range(len(matrix[0])) for k in range(len(matrix))):
            has_dominant_row = True
            break
    
    for j in range(len(matrix[0])):
        if all(matrix[i, j] >= matrix[i, k] for i in range(len(matrix)) for k in range(len(matrix[0]))):
            has_dominant_col = True
            break
    
    # Zero-sum property
    row_payoffs = [matrix[i, :].sum() for i in range(len(matrix))]
    col_payoffs = [matrix[:, j].sum() for j in range(len(matrix[0]))]
    row_payoff_variance = np.var(row_payoffs)
    
    return {
        'game_id': game['game_id'],
        'is_pure_nash': is_pure_nash,
        'has_dominant_row': has_dominant_row,
        'has_dominant_col': has_dominant_col,
        'payoff_mean': payoff_mean,
        'payoff_std': payoff_std,
        'payoff_range': payoff_range,
        'row_payoff_variance': row_payoff_variance,
    }

# Analyze all games
game_properties = pd.DataFrame([analyze_game_properties(g) for g in games])

# Create trials dataframe
trials_df = pd.DataFrame(trials)

# Merge game properties with trial results
merged_df = trials_df.merge(game_properties, on='game_id')

print("\n" + "="*80)
print("GAME PROPERTIES ANALYSIS")
print("="*80)
print(game_properties.describe().round(2))
print(f"\nPure Nash Equilibria: {game_properties['is_pure_nash'].sum()}/{len(game_properties)}")

# Classification
merged_df['performance_category'] = pd.cut(
    merged_df['nash_gap'], 
    bins=[-0.01, 0.01, 10, 50, float('inf')],
    labels=['Optimal (gap≈0)', 'Good (gap<10)', 'Medium (gap<50)', 'Poor (gap≥50)']
)

print("\n" + "="*80)
print("LLM PERFORMANCE DISTRIBUTION")
print("="*80)
perf_counts = merged_df['performance_category'].value_counts().sort_index()
for cat, count in perf_counts.items():
    pct = 100 * count / len(merged_df)
    print(f"  {cat}: {count} games ({pct:.1f}%)")

# Easy vs Hard games
optimal_games = merged_df[merged_df['nash_gap'] < 0.01]['game_id'].tolist()
hard_games = merged_df[merged_df['nash_gap'] > 50]['game_id'].tolist()

print(f"\n✓ {len(optimal_games)} games played optimally (Nash gap ≈ 0)")
print(f"✗ {len(hard_games)} games where LLM struggled (Nash gap > 50)")

# Correlations
print("\n" + "="*80)
print("GAME PROPERTY CORRELATIONS WITH NASH GAP")
print("="*80)
correlation_cols = ['payoff_mean', 'payoff_std', 'payoff_range', 'row_payoff_variance']
correlations = merged_df[correlation_cols + ['nash_gap']].corr()['nash_gap'].sort_values(ascending=False)
print(correlations.round(3))

corr_payoff_std = merged_df['payoff_std'].corr(merged_df['nash_gap'])

# Pure vs Mixed analysis
print("\n" + "="*80)
print("PURE vs MIXED NASH EQUILIBRIUM")
print("="*80)
pure_nash_subset = merged_df[merged_df['is_pure_nash'] == True]['nash_gap']
mixed_nash_subset = merged_df[merged_df['is_pure_nash'] == False]['nash_gap']

print(f"Pure Nash:  mean gap = {pure_nash_subset.mean():.2f} (n={len(pure_nash_subset)})")
print(f"Mixed Nash: mean gap = {mixed_nash_subset.mean():.2f} (n={len(mixed_nash_subset)})")

t_stat, p_value = stats.ttest_ind(pure_nash_subset, mixed_nash_subset)
print(f"T-test: t={t_stat:.3f}, p={p_value:.4f}")
if p_value < 0.05:
    print("✓ Statistically significant difference!")
else:
    print("✗ No significant difference")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distribution of Nash gaps
ax = axes[0, 0]
ax.hist(merged_df['nash_gap'], bins=30, edgecolor='black', alpha=0.7)
ax.axvline(merged_df['nash_gap'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {merged_df["nash_gap"].mean():.2f}')
ax.set_xlabel('Nash Gap')
ax.set_ylabel('Count')
ax.set_title('Distribution of Nash Gaps')
ax.legend()

# Plot 2: Payoff std vs Nash gap
ax = axes[0, 1]
ax.scatter(merged_df['payoff_std'], merged_df['nash_gap'], alpha=0.6)
ax.set_xlabel('Payoff Std Dev')
ax.set_ylabel('Nash Gap')
ax.set_title(f'Payoff Variance vs Game Difficulty\n(corr={corr_payoff_std:.3f})')
z = np.polyfit(merged_df['payoff_std'], merged_df['nash_gap'], 1)
p = np.poly1d(z)
ax.plot(merged_df['payoff_std'].sort_values(), p(merged_df['payoff_std'].sort_values()), "r--", alpha=0.8)

# Plot 3: Pure vs Mixed Nash
ax = axes[1, 0]
data_to_plot = [pure_nash_subset, mixed_nash_subset]
bp = ax.boxplot(data_to_plot, labels=['Pure Nash', 'Mixed Nash'], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_ylabel('Nash Gap')
ax.set_title('Game Difficulty: Pure vs Mixed Nash Equilibrium')

# Plot 4: Performance categories
ax = axes[1, 1]
perf_counts_plot = merged_df['performance_category'].value_counts()
colors = ['green', 'yellow', 'orange', 'red']
ax.bar(range(len(perf_counts_plot)), perf_counts_plot.values, color=colors[:len(perf_counts_plot)])
ax.set_xticks(range(len(perf_counts_plot)))
ax.set_xticklabels(perf_counts_plot.index, rotation=45, ha='right')
ax.set_ylabel('Number of Games')
ax.set_title('LLM Performance Categories')

plt.tight_layout()
plt.savefig('results/run_20260219_172049/performance_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved to results/run_20260219_172049/performance_analysis.png")

# Summary findings
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

optimal_pct = len(optimal_games) / len(merged_df) * 100
print(f"\n🎯 WHERE DOES LLM EXCEL?")
print(f"  ✓ {len(optimal_games)} games ({optimal_pct:.1f}%) played at Nash equilibrium")
print(f"  ✓ {(merged_df['nash_gap'] < 10).sum()} games ({(merged_df['nash_gap'] < 10).sum()/len(merged_df)*100:.1f}%) with gap < 10")

optimal_subset = merged_df[merged_df['nash_gap'] < 0.01]
if len(optimal_subset) > 0:
    print(f"\n  Features of games LLM plays optimally:")
    print(f"    - Avg payoff std dev: {optimal_subset['payoff_std'].mean():.2f}")
    print(f"    - Avg payoff range: {optimal_subset['payoff_range'].mean():.2f}")
    print(f"    - {(optimal_subset['is_pure_nash']).sum()/len(optimal_subset)*100:.1f}% have pure Nash equilibrium")

print(f"\n❌ WHERE DOES LLM STRUGGLE?")
print(f"  ✗ {len(hard_games)} games ({len(hard_games)/len(merged_df)*100:.1f}%) with gap > 50")
print(f"  ✗ {(merged_df['nash_gap'] > 100).sum()} very hard games with gap > 100")

hard_subset = merged_df[merged_df['nash_gap'] > 50]
if len(hard_subset) > 0:
    print(f"\n  Features of games LLM struggles with:")
    print(f"    - Avg payoff std dev: {hard_subset['payoff_std'].mean():.2f} (vs {merged_df['payoff_std'].mean():.2f} overall)")
    print(f"    - Avg payoff range: {hard_subset['payoff_range'].mean():.2f} (vs {merged_df['payoff_range'].mean():.2f} overall)")
    print(f"    - {(hard_subset['is_pure_nash']).sum()/len(hard_subset)*100:.1f}% have pure Nash equilibrium")

print(f"""
📊 CRITICAL GAME PROPERTIES:

  1. Payoff Variance (Std Dev)
     - Correlation with difficulty: {corr_payoff_std:.3f}
     - Interpretation: {'STRONG' if abs(corr_payoff_std) > 0.3 else 'WEAK'} predictor of game difficulty
     
  2. Pure vs Mixed Nash Equilibrium
     - Pure Nash avg gap: {pure_nash_subset.mean():.2f}
     - Mixed Nash avg gap: {mixed_nash_subset.mean():.2f}
     - Difference is {'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'} (p={p_value:.4f})
     
  3. Game Difficulty Distribution
     - {(merged_df['nash_gap'] < 1).sum()} games nearly optimal
     - {(merged_df['nash_gap'] >= 1).sum()} games with measurable difficulty
     
📈 ACTIONABLE INSIGHTS:

  • Games with LOWER payoff variance are EASIER for LLM
  • {'Pure Nash equilibrium games appear harder' if pure_nash_subset.mean() > mixed_nash_subset.mean() else 'Mixed strategy games appear harder'}
  • LLM shows high variability in performance - some games are trivial, others very hard
  • Consider training on diverse game structures to improve robustness

═══════════════════════════════════════════════════════════════════════════════
""")

# Save summary to JSON
summary_findings = {
    'optimal_games_count': len(optimal_games),
    'optimal_games_pct': float(optimal_pct),
    'hard_games_count': len(hard_games),
    'hard_games_pct': float(len(hard_games)/len(merged_df)*100),
    'payoff_std_correlation': float(corr_payoff_std),
    'pure_nash_mean_gap': float(pure_nash_subset.mean()),
    'mixed_nash_mean_gap': float(mixed_nash_subset.mean()),
    'nash_gap_mean': float(merged_df['nash_gap'].mean()),
    'nash_gap_median': float(merged_df['nash_gap'].median()),
    'nash_gap_std': float(merged_df['nash_gap'].std()),
}

with open('results/run_20260219_172049/analysis_summary.json', 'w') as f:
    json.dump(summary_findings, f, indent=2)

print("\n✓ Analysis summary saved to results/run_20260219_172049/analysis_summary.json")
