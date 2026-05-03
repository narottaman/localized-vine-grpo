#!/usr/bin/env python3
"""
Compare Base Model vs VinePPO Results
Generates paper-ready comparison tables and plots
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_metrics(path: str) -> dict:
    """Load metrics JSON"""
    with open(path, 'r') as f:
        return json.load(f)


def generate_comparison_table(base_metrics: dict, vineppo_metrics: dict) -> str:
    """Generate formatted comparison table"""
    
    table = []
    table.append("=" * 80)
    table.append("📊 BASE MODEL vs VinePPO - COMPARISON RESULTS")
    table.append("=" * 80)
    table.append("")
    
    # Core metrics
    table.append("🎯 CORE ACCURACY METRICS")
    table.append("-" * 80)
    table.append(f"{'Metric':<40} {'Base Model':>15} {'VinePPO':>15} {'Δ':>10}")
    table.append("-" * 80)
    
    metrics_to_compare = [
        ("Perfect Solve Rate", "perfect_solve_rate", "%"),
        ("Mean Cell Accuracy", "mean_cell_accuracy", ""),
        ("Overall Cell Accuracy", "overall_cell_accuracy", ""),
        ("Median Cell Accuracy", "median_cell_accuracy", ""),
    ]
    
    for label, key, unit in metrics_to_compare:
        base_val = base_metrics[key]
        vineppo_val = vineppo_metrics[key]
        delta = vineppo_val - base_val
        
        if unit == "%":
            base_str = f"{base_val*100:.1f}%"
            vineppo_str = f"{vineppo_val*100:.1f}%"
            delta_str = f"{delta*100:+.1f}pp"
        else:
            base_str = f"{base_val:.3f}"
            vineppo_str = f"{vineppo_val:.3f}"
            delta_str = f"{delta:+.3f}"
        
        table.append(f"{label:<40} {base_str:>15} {vineppo_str:>15} {delta_str:>10}")
    
    table.append("")
    
    # Reasoning quality
    table.append("🧠 REASONING QUALITY METRICS")
    table.append("-" * 80)
    table.append(f"{'Metric':<40} {'Base Model':>15} {'VinePPO':>15} {'Δ':>10}")
    table.append("-" * 80)
    
    reasoning_metrics = [
        ("Structured Reasoning Rate", "structured_reasoning_rate", "%"),
        ("Explicit Deduction Rate", "explicit_deduction_rate", "%"),
        ("Mean Reasoning Steps", "mean_num_steps", ""),
        ("Mean Constraints Mentioned", "mean_constraints_mentioned", ""),
    ]
    
    for label, key, unit in reasoning_metrics:
        base_val = base_metrics[key]
        vineppo_val = vineppo_metrics[key]
        delta = vineppo_val - base_val
        
        if unit == "%":
            base_str = f"{base_val*100:.1f}%"
            vineppo_str = f"{vineppo_val*100:.1f}%"
            delta_str = f"{delta*100:+.1f}pp"
        else:
            base_str = f"{base_val:.2f}"
            vineppo_str = f"{vineppo_val:.2f}"
            delta_str = f"{delta:+.2f}"
        
        table.append(f"{label:<40} {base_str:>15} {vineppo_str:>15} {delta_str:>10}")
    
    table.append("")
    
    # Difficulty breakdown
    table.append("📏 PERFORMANCE BY PUZZLE DIFFICULTY")
    table.append("-" * 80)
    
    for difficulty in ['small', 'medium', 'large']:
        table.append(f"\n{difficulty.upper()} Puzzles:")
        
        base_acc = base_metrics['by_difficulty'][difficulty]['accuracy']
        vineppo_acc = vineppo_metrics['by_difficulty'][difficulty]['accuracy']
        base_perf = base_metrics['by_difficulty'][difficulty]['perfect_rate']
        vineppo_perf = vineppo_metrics['by_difficulty'][difficulty]['perfect_rate']
        
        table.append(f"  Accuracy:      Base={base_acc:.3f}, VinePPO={vineppo_acc:.3f}, Δ={vineppo_acc-base_acc:+.3f}")
        table.append(f"  Perfect Rate:  Base={base_perf:.1%}, VinePPO={vineppo_perf:.1%}, Δ={vineppo_perf-base_perf:+.1%}")
    
    table.append("")
    table.append("=" * 80)
    
    # Statistical significance note
    table.append("\n📊 INTERPRETATION NOTES:")
    table.append("-" * 80)
    
    perfect_delta = vineppo_metrics['perfect_solve_rate'] - base_metrics['perfect_solve_rate']
    acc_delta = vineppo_metrics['mean_cell_accuracy'] - base_metrics['mean_cell_accuracy']
    
    if perfect_delta > 0.05:
        table.append(f"✅ VinePPO shows SIGNIFICANT improvement in perfect solve rate (+{perfect_delta*100:.1f}pp)")
    elif perfect_delta > 0:
        table.append(f"📈 VinePPO shows modest improvement in perfect solve rate (+{perfect_delta*100:.1f}pp)")
    else:
        table.append(f"⚠️  VinePPO shows degradation in perfect solve rate ({perfect_delta*100:.1f}pp)")
    
    if acc_delta > 0.02:
        table.append(f"✅ VinePPO shows SIGNIFICANT improvement in cell accuracy (+{acc_delta:.3f})")
    elif acc_delta > 0:
        table.append(f"📈 VinePPO shows modest improvement in cell accuracy (+{acc_delta:.3f})")
    else:
        table.append(f"⚠️  VinePPO shows degradation in cell accuracy ({acc_delta:.3f})")
    
    table.append("")
    table.append("For paper reporting, focus on:")
    table.append("  1. Perfect Solve Rate (primary metric for logic puzzles)")
    table.append("  2. Mean Cell Accuracy (secondary metric)")
    table.append("  3. Performance breakdown by difficulty (shows generalization)")
    table.append("  4. Reasoning quality improvements (shows better process)")
    table.append("")
    table.append("=" * 80)
    
    return "\n".join(table)


def generate_comparison_plots(base_metrics: dict, vineppo_metrics: dict, output_path: str):
    """Generate side-by-side comparison plots"""
    
    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Base Model vs VinePPO Comparison', fontsize=16, fontweight='bold')
    
    # 1. Perfect Solve Rate Comparison
    models = ['Base\nModel', 'VinePPO']
    perfect_rates = [
        base_metrics['perfect_solve_rate'] * 100,
        vineppo_metrics['perfect_solve_rate'] * 100
    ]
    
    bars = axes[0, 0].bar(models, perfect_rates, color=['#4a90e2', '#50c878'], alpha=0.8, edgecolor='black')
    axes[0, 0].set_ylabel('Perfect Solve Rate (%)')
    axes[0, 0].set_title('Perfect Solve Rate', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add delta annotation
    delta = perfect_rates[1] - perfect_rates[0]
    axes[0, 0].annotate(f'Δ = {delta:+.1f}pp',
                        xy=(0.5, max(perfect_rates) * 0.9),
                        ha='center', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # 2. Mean Cell Accuracy Comparison
    accuracies = [
        base_metrics['mean_cell_accuracy'],
        vineppo_metrics['mean_cell_accuracy']
    ]
    
    bars = axes[0, 1].bar(models, accuracies, color=['#4a90e2', '#50c878'], alpha=0.8, edgecolor='black')
    axes[0, 1].set_ylabel('Mean Cell Accuracy')
    axes[0, 1].set_title('Mean Cell Accuracy', fontweight='bold')
    axes[0, 1].set_ylim([0, 1.0])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    delta = accuracies[1] - accuracies[0]
    axes[0, 1].annotate(f'Δ = {delta:+.3f}',
                        xy=(0.5, max(accuracies) * 0.9),
                        ha='center', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # 3. Performance by Difficulty
    difficulties = ['Small', 'Medium', 'Large']
    base_acc_by_diff = [
        base_metrics['by_difficulty']['small']['accuracy'],
        base_metrics['by_difficulty']['medium']['accuracy'],
        base_metrics['by_difficulty']['large']['accuracy']
    ]
    vineppo_acc_by_diff = [
        vineppo_metrics['by_difficulty']['small']['accuracy'],
        vineppo_metrics['by_difficulty']['medium']['accuracy'],
        vineppo_metrics['by_difficulty']['large']['accuracy']
    ]
    
    x = np.arange(len(difficulties))
    width = 0.35
    
    bars1 = axes[1, 0].bar(x - width/2, base_acc_by_diff, width, 
                           label='Base Model', color='#4a90e2', alpha=0.8, edgecolor='black')
    bars2 = axes[1, 0].bar(x + width/2, vineppo_acc_by_diff, width,
                           label='VinePPO', color='#50c878', alpha=0.8, edgecolor='black')
    
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy by Puzzle Difficulty', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(difficulties)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 1.0])
    
    # 4. Reasoning Quality Comparison
    reasoning_metrics_names = ['Structured\nReasoning', 'Explicit\nDeduction', 'Mean\nSteps']
    base_reasoning = [
        base_metrics['structured_reasoning_rate'] * 100,
        base_metrics['explicit_deduction_rate'] * 100,
        base_metrics['mean_num_steps']
    ]
    vineppo_reasoning = [
        vineppo_metrics['structured_reasoning_rate'] * 100,
        vineppo_metrics['explicit_deduction_rate'] * 100,
        vineppo_metrics['mean_num_steps']
    ]
    
    # Normalize for comparison (scale steps to 0-100 range)
    base_reasoning[2] = (base_reasoning[2] / max(base_reasoning[2], vineppo_reasoning[2])) * 100
    vineppo_reasoning[2] = (vineppo_reasoning[2] / max(base_reasoning[2], vineppo_reasoning[2])) * 100
    
    x = np.arange(len(reasoning_metrics_names))
    bars1 = axes[1, 1].bar(x - width/2, base_reasoning, width,
                           label='Base Model', color='#4a90e2', alpha=0.8, edgecolor='black')
    bars2 = axes[1, 1].bar(x + width/2, vineppo_reasoning, width,
                           label='VinePPO', color='#50c878', alpha=0.8, edgecolor='black')
    
    axes[1, 1].set_ylabel('Score (normalized to %)')
    axes[1, 1].set_title('Reasoning Quality Metrics', fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(reasoning_metrics_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison plots saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare Base vs VinePPO results')
    parser.add_argument('--base_results', type=str, required=True, help='Path to base model metrics.json')
    parser.add_argument('--vineppo_results', type=str, required=True, help='Path to VinePPO metrics.json')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Load metrics
    print("📊 Loading metrics...")
    base_metrics = load_metrics(args.base_results)
    vineppo_metrics = load_metrics(args.vineppo_results)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate comparison table
    print("📝 Generating comparison report...")
    comparison_text = generate_comparison_table(base_metrics, vineppo_metrics)
    
    # Save report
    report_path = Path(args.output_dir) / "comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write(comparison_text)
    
    print(comparison_text)
    print(f"\n✅ Report saved to {report_path}")
    
    # Generate plots
    print("\n📊 Generating comparison plots...")
    plot_path = str(Path(args.output_dir) / "comparison_plots.png")
    generate_comparison_plots(base_metrics, vineppo_metrics, plot_path)
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()