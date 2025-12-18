"""
Analyze and visualize experimental results.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_results(result_file):
    """Load results from JSON file."""
    with open(result_file, 'r') as f:
        return json.load(f)


def plot_comparison(cross_domain_results, baseline_results, output_dir):
    """Plot comparison between cross-domain and baseline."""
    
    # Extract metrics
    tasks = list(cross_domain_results.keys())
    
    cd_maes = [cross_domain_results[task]["energy_mae"] for task in tasks]
    bl_maes = [baseline_results[task]["energy_mae"] for task in tasks]
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(tasks))
    width = 0.35
    
    ax.bar(x - width/2, bl_maes, width, label='Baseline', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, cd_maes, width, label='Cross-Domain', color='#27ae60', alpha=0.8)
    
    ax.set_xlabel('GeoShift Tasks', fontsize=14)
    ax.set_ylabel('Energy MAE (eV)', fontsize=14)
    ax.set_title('Cross-Domain vs Baseline Performance', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([task.replace('_', ' ').title() for task in tasks], rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=300)
    print(f"Saved comparison plot to {output_dir}/comparison.png")
    
    # Compute improvements
    improvements = [(bl - cd) / bl * 100 for bl, cd in zip(bl_maes, cd_maes)]
    avg_improvement = np.mean(improvements)
    
    print(f"\nAverage improvement: {avg_improvement:.1f}%")
    for task, imp in zip(tasks, improvements):
        print(f"  {task}: {imp:.1f}%")


def plot_sample_efficiency(train_sizes, cd_scores, bl_scores, output_dir):
    """Plot sample efficiency curves."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train_sizes, bl_scores, 'o-', label='Baseline', color='#e74c3c', linewidth=2)
    ax.plot(train_sizes, cd_scores, 'o-', label='Cross-Domain', color='#27ae60', linewidth=2)
    
    ax.set_xlabel('Training Samples', fontsize=14)
    ax.set_ylabel('Test MAE (eV)', fontsize=14)
    ax.set_title('Sample Efficiency', fontsize=16)
    ax.set_xscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_efficiency.png'), dpi=300)
    print(f"Saved sample efficiency plot to {output_dir}/sample_efficiency.png")


def generate_report(cross_domain_results, baseline_results, output_dir):
    """Generate markdown report."""
    
    report = "# Experimental Results\n\n"
    report += "## GeoShift Benchmark Results\n\n"
    report += "| Task | Baseline MAE | Cross-Domain MAE | Improvement |\n"
    report += "|------|--------------|------------------|-------------|\n"
    
    for task in cross_domain_results.keys():
        bl_mae = baseline_results[task]["energy_mae"]
        cd_mae = cross_domain_results[task]["energy_mae"]
        improvement = (bl_mae - cd_mae) / bl_mae * 100
        
        report += f"| {task.replace('_', ' ').title()} | {bl_mae:.4f} | {cd_mae:.4f} | **{improvement:.1f}%** |\n"
    
    # Overall statistics
    bl_maes = [baseline_results[task]["energy_mae"] for task in cross_domain_results.keys()]
    cd_maes = [cross_domain_results[task]["energy_mae"] for task in cross_domain_results.keys()]
    
    avg_improvement = np.mean([(bl - cd) / bl * 100 for bl, cd in zip(bl_maes, cd_maes)])
    
    report += f"\n**Average Improvement:** {avg_improvement:.1f}%\n\n"
    
    # Save report
    with open(os.path.join(output_dir, 'RESULTS.md'), 'w') as f:
        f.write(report)
    
    print(f"Saved report to {output_dir}/RESULTS.md")


def main():
    parser = argparse.ArgumentParser(description="Analyze experimental results")
    parser.add_argument("--cross-domain", type=str, required=True,
                        help="Path to cross-domain results JSON")
    parser.add_argument("--baseline", type=str, required=True,
                        help="Path to baseline results JSON")
    parser.add_argument("--output", type=str, default="./docs/images",
                        help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load results
    print("Loading results...")
    cd_results = load_results(args.cross_domain)
    bl_results = load_results(args.baseline)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_comparison(cd_results, bl_results, args.output)
    
    # Generate report
    print("\nGenerating report...")
    generate_report(cd_results, bl_results, args.output)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
