"""
Visualization Module

Creates comprehensive dashboards and plots for safety evaluation results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (20, 15)


class EvaluationVisualizer:
    """Creates visualizations for safety evaluation results."""

    def __init__(self, style: str = "whitegrid"):
        """
        Initialize visualizer.

        Args:
            style: Seaborn style theme
        """
        sns.set_theme(style=style)

    def create_dashboard(self,
                        evaluated_df: pd.DataFrame,
                        output_path: str = "data/results/evaluation_dashboard.png",
                        figsize: tuple = (20, 15)) -> None:
        """
        Create comprehensive 9-subplot evaluation dashboard.

        Args:
            evaluated_df: DataFrame with evaluation results
            output_path: Path to save the dashboard
            figsize: Figure size (width, height)
        """
        logger.info("Creating evaluation dashboard")

        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle('LLM Safety Evaluation Dashboard', fontsize=20, fontweight='bold')

        # 1. Toxicity distribution by category
        self._plot_toxicity_distribution(evaluated_df, axes[0, 0])

        # 2. Box plot of toxicity scores
        self._plot_toxicity_boxplot(evaluated_df, axes[0, 1])

        # 3. Toxicity metrics heatmap
        self._plot_toxicity_heatmap(evaluated_df, axes[0, 2])

        # 4. Category comparison (mean scores)
        self._plot_category_comparison(evaluated_df, axes[1, 0])

        # 5. Severe toxicity by category
        self._plot_severe_toxicity(evaluated_df, axes[1, 1])

        # 6. Threat and insult comparison
        self._plot_threat_insult(evaluated_df, axes[1, 2])

        # 7. Identity attack distribution
        self._plot_identity_attack(evaluated_df, axes[2, 0])

        # 8. Obscene content by category
        self._plot_obscene_content(evaluated_df, axes[2, 1])

        # 9. Overall safety score
        self._plot_overall_safety(evaluated_df, axes[2, 2])

        plt.tight_layout()

        # Save dashboard
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard saved to {output_path}")

        plt.close()

    def _plot_toxicity_distribution(self, df: pd.DataFrame, ax):
        """Plot toxicity score distribution by category."""
        for category in df['category'].unique():
            category_data = df[df['category'] == category]['toxicity']
            ax.hist(category_data, alpha=0.6, label=category, bins=20)

        ax.set_xlabel('Toxicity Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Toxicity Score Distribution by Category')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_toxicity_boxplot(self, df: pd.DataFrame, ax):
        """Plot boxplot of toxicity scores."""
        sns.boxplot(data=df, x='category', y='toxicity', ax=ax, palette='Set2')
        ax.set_xlabel('Category')
        ax.set_ylabel('Toxicity Score')
        ax.set_title('Toxicity Score Distribution (Box Plot)')
        ax.tick_params(axis='x', rotation=45)

    def _plot_toxicity_heatmap(self, df: pd.DataFrame, ax):
        """Plot heatmap of mean toxicity metrics."""
        toxicity_cols = ['toxicity', 'severe_toxicity', 'obscene',
                        'threat', 'insult', 'identity_attack']
        heatmap_data = df.groupby('category')[toxicity_cols].mean()

        sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='YlOrRd',
                   ax=ax, cbar_kws={'label': 'Mean Score'})
        ax.set_title('Mean Toxicity Metrics by Category')
        ax.set_xlabel('Category')
        ax.set_ylabel('Metric')

    def _plot_category_comparison(self, df: pd.DataFrame, ax):
        """Plot category comparison bar chart."""
        toxicity_cols = ['toxicity', 'severe_toxicity', 'obscene',
                        'threat', 'insult', 'identity_attack']
        category_means = df.groupby('category')[toxicity_cols].mean()

        category_means.plot(kind='bar', ax=ax, width=0.8)
        ax.set_xlabel('Category')
        ax.set_ylabel('Mean Score')
        ax.set_title('Mean Toxicity Metrics by Category')
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    def _plot_severe_toxicity(self, df: pd.DataFrame, ax):
        """Plot severe toxicity distribution."""
        sns.violinplot(data=df, x='category', y='severe_toxicity',
                      ax=ax, palette='muted')
        ax.set_xlabel('Category')
        ax.set_ylabel('Severe Toxicity Score')
        ax.set_title('Severe Toxicity Distribution')
        ax.tick_params(axis='x', rotation=45)

    def _plot_threat_insult(self, df: pd.DataFrame, ax):
        """Plot threat vs insult comparison."""
        category_data = df.groupby('category')[['threat', 'insult']].mean()
        category_data.plot(kind='bar', ax=ax, color=['#ff9999', '#66b3ff'])
        ax.set_xlabel('Category')
        ax.set_ylabel('Mean Score')
        ax.set_title('Threat vs Insult Scores')
        ax.legend(title='Type')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    def _plot_identity_attack(self, df: pd.DataFrame, ax):
        """Plot identity attack distribution."""
        sns.kdeplot(data=df, x='identity_attack', hue='category',
                   ax=ax, fill=True, alpha=0.5)
        ax.set_xlabel('Identity Attack Score')
        ax.set_ylabel('Density')
        ax.set_title('Identity Attack Score Distribution')
        ax.legend(title='Category')

    def _plot_obscene_content(self, df: pd.DataFrame, ax):
        """Plot obscene content by category."""
        category_obscene = df.groupby('category')['obscene'].mean()
        colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(category_obscene)))
        category_obscene.plot(kind='bar', ax=ax, color=colors)
        ax.set_xlabel('Category')
        ax.set_ylabel('Mean Obscene Score')
        ax.set_title('Obscene Content by Category')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    def _plot_overall_safety(self, df: pd.DataFrame, ax):
        """Plot overall safety score (inverse of mean toxicity)."""
        toxicity_cols = ['toxicity', 'severe_toxicity', 'obscene',
                        'threat', 'insult', 'identity_attack']
        overall_toxicity = df.groupby('category')[toxicity_cols].mean().mean(axis=1)
        overall_safety = 1 - overall_toxicity

        colors = ['green' if x > 0.7 else 'orange' if x > 0.5 else 'red'
                 for x in overall_safety]

        overall_safety.plot(kind='barh', ax=ax, color=colors)
        ax.set_xlabel('Safety Score (1 - Mean Toxicity)')
        ax.set_ylabel('Category')
        ax.set_title('Overall Safety Score by Category')
        ax.set_xlim([0, 1])
        ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Good')
        ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_comparison(self,
                       baseline_df: pd.DataFrame,
                       improved_df: pd.DataFrame,
                       output_path: str = "data/results/comparison.png"):
        """
        Create comparison plot between baseline and improved model.

        Args:
            baseline_df: Baseline evaluation results
            improved_df: Improved model evaluation results
            output_path: Path to save the plot
        """
        logger.info("Creating comparison plot")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Baseline vs Improved Model Comparison', fontsize=16, fontweight='bold')

        toxicity_cols = ['toxicity', 'severe_toxicity', 'obscene',
                        'threat', 'insult', 'identity_attack']

        # Overall comparison
        baseline_mean = baseline_df[toxicity_cols].mean()
        improved_mean = improved_df[toxicity_cols].mean()

        comparison_df = pd.DataFrame({
            'Baseline': baseline_mean,
            'Improved': improved_mean
        })

        comparison_df.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Overall Metric Comparison')
        axes[0, 0].set_ylabel('Mean Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Category-wise comparison
        for idx, category in enumerate(['toxicity', 'stereotype', 'privacy']):
            if idx < 3:
                row, col = (idx + 1) // 2, (idx + 1) % 2
                baseline_cat = baseline_df[baseline_df['category'] == category][toxicity_cols].mean()
                improved_cat = improved_df[improved_df['category'] == category][toxicity_cols].mean()

                cat_comparison = pd.DataFrame({
                    'Baseline': baseline_cat,
                    'Improved': improved_cat
                })

                cat_comparison.plot(kind='bar', ax=axes[row, col])
                axes[row, col].set_title(f'{category.capitalize()} Category')
                axes[row, col].set_ylabel('Mean Score')
                axes[row, col].tick_params(axis='x', rotation=45)
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {output_path}")
        plt.close()


def create_dashboard(evaluated_df: pd.DataFrame,
                    output_path: str = "data/results/evaluation_dashboard.png") -> None:
    """
    Convenience function to create evaluation dashboard.

    Args:
        evaluated_df: DataFrame with evaluation results
        output_path: Path to save the dashboard
    """
    visualizer = EvaluationVisualizer()
    visualizer.create_dashboard(evaluated_df, output_path)


if __name__ == "__main__":
    print("Visualization Module")
    print("Use this module to create evaluation dashboards and plots.")
