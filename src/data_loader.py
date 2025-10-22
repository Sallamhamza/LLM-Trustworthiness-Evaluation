"""
Data Loading and Sampling Module

Handles loading benchmark datasets (DecodingTrust, TrustLLM) and
performing stratified sampling for balanced evaluation.
"""

from datasets import load_dataset
import pandas as pd
import random
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkDataLoader:
    """Loads and samples from safety benchmark datasets."""

    def __init__(self, random_seed: int = 42):
        """
        Initialize data loader.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)

    def load_decodingtrust(self,
                          categories: List[str] = ["toxicity", "stereotype", "privacy"],
                          samples_per_category: int = 200) -> pd.DataFrame:
        """
        Load DecodingTrust benchmark datasets.

        Args:
            categories: List of categories to load
            samples_per_category: Number of samples per category

        Returns:
            DataFrame with columns: prompt, category, split
        """
        logger.info(f"Loading DecodingTrust datasets: {categories}")

        all_prompts = []

        try:
            # Load DecodingTrust dataset
            dataset = load_dataset("AI-Secure/DecodingTrust", trust_remote_code=True)

            for category in categories:
                category_prompts = self._sample_category(
                    dataset,
                    category,
                    samples_per_category
                )
                all_prompts.extend(category_prompts)

            df = pd.DataFrame(all_prompts)
            logger.info(f"Loaded {len(df)} total prompts")
            logger.info(f"Distribution:\n{df['category'].value_counts()}")

            return df

        except Exception as e:
            logger.error(f"Error loading DecodingTrust: {e}")
            raise

    def _sample_category(self,
                        dataset,
                        category: str,
                        num_samples: int) -> List[Dict]:
        """
        Sample prompts from a specific category with stratification.

        Args:
            dataset: Loaded dataset object
            category: Category name
            num_samples: Number of samples to extract

        Returns:
            List of dictionaries with prompt data
        """
        prompts = []

        # Get available splits for this category
        category_data = dataset.get(category, {})

        if not category_data:
            logger.warning(f"Category {category} not found in dataset")
            return prompts

        # Collect all available prompts from all splits
        all_category_prompts = []
        for split_name, split_data in category_data.items():
            for item in split_data:
                all_category_prompts.append({
                    'prompt': item.get('prompt', item.get('text', '')),
                    'category': category,
                    'split': split_name
                })

        # Stratified sampling across splits
        if len(all_category_prompts) <= num_samples:
            prompts = all_category_prompts
        else:
            prompts = random.sample(all_category_prompts, num_samples)

        logger.info(f"Sampled {len(prompts)} prompts from {category}")
        return prompts

    def load_custom_prompts(self, file_path: str) -> pd.DataFrame:
        """
        Load prompts from a custom CSV or JSON file.

        Args:
            file_path: Path to the file

        Returns:
            DataFrame with prompts
        """
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")

        logger.info(f"Loaded {len(df)} custom prompts from {file_path}")
        return df


def load_benchmark_datasets(num_samples: int = 600,
                            categories: List[str] = ["toxicity", "stereotype", "privacy"],
                            random_seed: int = 42) -> pd.DataFrame:
    """
    Convenience function to load benchmark datasets.

    Args:
        num_samples: Total number of samples (divided equally across categories)
        categories: List of categories to evaluate
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with prompts
    """
    samples_per_category = num_samples // len(categories)

    loader = BenchmarkDataLoader(random_seed=random_seed)
    return loader.load_decodingtrust(
        categories=categories,
        samples_per_category=samples_per_category
    )


def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for loaded prompts.

    Args:
        df: DataFrame with prompts

    Returns:
        Summary DataFrame
    """
    summary = df.groupby(['category', 'split']).size().reset_index(name='count')
    return summary


if __name__ == "__main__":
    # Example usage
    print("Loading benchmark datasets...")
    prompts_df = load_benchmark_datasets(num_samples=600)

    print(f"\nTotal prompts: {len(prompts_df)}")
    print("\nCategory distribution:")
    print(prompts_df['category'].value_counts())

    print("\nDetailed summary:")
    print(get_category_summary(prompts_df))

    print("\nSample prompts:")
    print(prompts_df.head())
