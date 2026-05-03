#!/usr/bin/env python3
"""
Fix already-computed results by loading CSV and regenerating metrics.json
Use this if evaluation completed but JSON saving failed.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def convert_to_native(obj):
    """Convert numpy/pandas types to native Python types"""
    if isinstance(obj, (np.integer, pd.Int64Dtype)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        # Handle NaN
        if pd.isna(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj


def compute_aggregate_metrics(df: pd.DataFrame) -> dict:
    """Compute aggregate metrics from results dataframe"""
    
    metrics = {
        'mean_cell_accuracy': df['accuracy'].mean(),
        'std_cell_accuracy': df['accuracy'].std(),
        'median_cell_accuracy': df['accuracy'].median(),
        'perfect_solve_rate': df['is_perfect'].mean(),
        'num_perfect_solves': int(df['is_perfect'].sum()),
        'total_puzzles': len(df),
        'total_cells': int(df['total_cells'].sum()),
        'total_correct_cells': int(df['correct_cells'].sum()),
        'overall_cell_accuracy': df['correct_cells'].sum() / df['total_cells'].sum(),
        'mean_reasoning_length': df['reasoning_length'].mean(),
        'mean_num_steps': df['num_steps'].mean(),
        'structured_reasoning_rate': df['has_structured_reasoning'].mean(),
        'explicit_deduction_rate': df['has_explicit_deduction'].mean(),
        'mean_constraints_mentioned': df['num_constraints_mentioned'].mean(),
        'accuracy_quartiles': {
            'q25': df['accuracy'].quantile(0.25),
            'q50': df['accuracy'].quantile(0.50),
            'q75': df['accuracy'].quantile(0.75),
        },
        'by_difficulty': {
            'small': {
                'accuracy': df[df['total_cells'] < 20]['accuracy'].mean() if len(df[df['total_cells'] < 20]) > 0 else 0.0,
                'perfect_rate': df[df['total_cells'] < 20]['is_perfect'].mean() if len(df[df['total_cells'] < 20]) > 0 else 0.0
            },
            'medium': {
                'accuracy': df[(df['total_cells'] >= 20) & (df['total_cells'] < 40)]['accuracy'].mean() if len(df[(df['total_cells'] >= 20) & (df['total_cells'] < 40)]) > 0 else 0.0,
                'perfect_rate': df[(df['total_cells'] >= 20) & (df['total_cells'] < 40)]['is_perfect'].mean() if len(df[(df['total_cells'] >= 20) & (df['total_cells'] < 40)]) > 0 else 0.0
            },
            'large': {
                'accuracy': df[df['total_cells'] >= 40]['accuracy'].mean() if len(df[df['total_cells'] >= 40]) > 0 else 0.0,
                'perfect_rate': df[df['total_cells'] >= 40]['is_perfect'].mean() if len(df[df['total_cells'] >= 40]) > 0 else 0.0
            }
        }
    }
    
    return metrics


def generate_visualizations(df: pd.DataFrame, output_dir: str):
    """Generate visualizations"""
    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Puzzle Baron Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. Accuracy distribution
    axes[0, 0].hist(df['accuracy'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['accuracy'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["accuracy"].mean():.3f}')
    axes[0, 0].set_xlabel('Cell Accuracy')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Cell Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Perfect solve rate
    perfect_counts = df['is_perfect'].value_counts()
    axes[0, 1].bar(['Incorrect', 'Perfect'], 
                   [perfect_counts.get(False, 0), perfect_counts.get(True, 0)],
                   color=['#ff6b6b', '#51cf66'])
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Perfect Solve Rate: {df["is_perfect"].mean():.1%}')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Accuracy vs puzzle size
    axes[0, 2].scatter(df['total_cells'], df['accuracy'], alpha=0.5)
    axes[0, 2].set_xlabel('Puzzle Size (# cells)')
    axes[0, 2].set_ylabel('Cell Accuracy')
    axes[0, 2].set_title('Accuracy vs Puzzle Complexity')
    if len(df) > 1:
        z = np.polyfit(df['total_cells'], df['accuracy'], 1)
        p = np.poly1d(z)
        axes[0, 2].plot(df['total_cells'].sort_values(), 
                        p(df['total_cells'].sort_values()), 
                        "r--", alpha=0.8, label='Trend')
        axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Reasoning steps
    axes[1, 0].hist(df['num_steps'], bins=15, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Number of Reasoning Steps')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Reasoning Step Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Accuracy by difficulty
    df['difficulty'] = pd.cut(df['total_cells'], bins=[0, 20, 40, 100], 
                               labels=['Small', 'Medium', 'Large'])
    difficulty_acc = df.groupby('difficulty')['accuracy'].mean()
    axes[1, 1].bar(difficulty_acc.index, difficulty_acc.values, 
                   color=['#51cf66', '#ffd43b', '#ff6b6b'])
    axes[1, 1].set_ylabel('Mean Accuracy')
    axes[1, 1].set_title('Accuracy by Puzzle Difficulty')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. Cumulative accuracy
    sorted_acc = np.sort(df['accuracy'])
    cumulative = np.arange(1, len(sorted_acc) + 1) / len(sorted_acc)
    axes[1, 2].plot(sorted_acc, cumulative, linewidth=2)
    axes[1, 2].set_xlabel('Cell Accuracy')
    axes[1, 2].set_ylabel('Cumulative Proportion')
    axes[1, 2].set_title('Cumulative Accuracy Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(0.5, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plots saved to {output_dir}/evaluation_plots.png")


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_results.py <output_dir>")
        print("Example: python fix_results.py /scratch/ngangada/thesis/thesis/eval_results/vineppo_step230")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    print("=" * 80)
    print("🔧 FIXING EVALUATION RESULTS")
    print("=" * 80)
    
    # Load CSV
    csv_path = os.path.join(output_dir, 'detailed_results.csv')
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found!")
        print(f"   Make sure the evaluation completed and saved the CSV file.")
        sys.exit(1)
    
    print(f"📊 Loading results from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   ✅ Loaded {len(df)} results")
    
    # Compute metrics
    print("📈 Computing metrics...")
    metrics = compute_aggregate_metrics(df)
    metrics_serializable = convert_to_native(metrics)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    print(f"💾 Saving metrics to {metrics_path}")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print("   ✅ Metrics saved")
    
    # Generate plots
    print("📊 Generating plots...")
    generate_visualizations(df, output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print(f"Perfect Solve Rate: {metrics['perfect_solve_rate']:.1%} ({metrics['num_perfect_solves']}/{metrics['total_puzzles']})")
    print(f"Mean Cell Accuracy: {metrics['mean_cell_accuracy']:.3f} ± {metrics['std_cell_accuracy']:.3f}")
    print(f"Overall Cell Accuracy: {metrics['overall_cell_accuracy']:.3f}")
    print(f"Median Accuracy: {metrics['median_cell_accuracy']:.3f}")
    print("\nReasoning Quality:")
    print(f"  Structured Reasoning Rate: {metrics['structured_reasoning_rate']:.1%}")
    print(f"  Explicit Deduction Rate: {metrics['explicit_deduction_rate']:.1%}")
    print(f"  Mean Steps: {metrics['mean_num_steps']:.1f}")
    print("=" * 80)
    
    print("\n✅ COMPLETE! Results saved to:")
    print(f"   - {metrics_path}")
    print(f"   - {os.path.join(output_dir, 'evaluation_plots.png')}")


if __name__ == "__main__":
    main()