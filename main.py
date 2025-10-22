"""
Main script for running LLM safety evaluation.

This script provides a command-line interface for the evaluation framework.
"""

import argparse
import json
import logging
from pathlib import Path

from src import (
    load_model,
    load_benchmark_datasets,
    generate_responses,
    evaluate_safety,
    create_dashboard,
    get_device_info
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="LLM Safety Evaluation Framework"
    )
    parser.add_argument(
        '--model-config',
        type=str,
        default='config/model_config.json',
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--eval-config',
        type=str,
        default='config/evaluation_config.json',
        help='Path to evaluation configuration file'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip generation and load existing responses'
    )
    parser.add_argument(
        '--device-info',
        action='store_true',
        help='Display device information and exit'
    )

    args = parser.parse_args()

    # Display device info if requested
    if args.device_info:
        device_info = get_device_info()
        logger.info("Device Information:")
        for key, value in device_info.items():
            logger.info(f"  {key}: {value}")
        return

    # Load configurations
    logger.info("Loading configurations...")
    model_config = load_config(args.model_config)
    eval_config = load_config(args.eval_config)

    # Override num_samples if provided
    if args.num_samples:
        eval_config['dataset']['total_samples'] = args.num_samples

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load Model
    logger.info("=" * 50)
    logger.info("Step 1: Loading model...")
    logger.info("=" * 50)
    model, tokenizer = load_model(model_config['model_name'])

    # Step 2: Load Datasets
    logger.info("=" * 50)
    logger.info("Step 2: Loading benchmark datasets...")
    logger.info("=" * 50)
    prompts_df = load_benchmark_datasets(
        num_samples=eval_config['dataset']['total_samples'],
        categories=eval_config['dataset']['categories'],
        random_seed=eval_config['dataset']['random_seed']
    )
    logger.info(f"Loaded {len(prompts_df)} prompts")
    logger.info(f"Category distribution:\n{prompts_df['category'].value_counts()}")

    # Step 3: Generate Responses
    if not args.skip_generation:
        logger.info("=" * 50)
        logger.info("Step 3: Generating responses...")
        logger.info("=" * 50)
        responses_df = generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompts_df=prompts_df,
            batch_size=eval_config['generation']['batch_size'],
            checkpoint_interval=eval_config['generation']['checkpoint_interval']
        )

        # Save responses
        responses_path = output_dir / 'responses.csv'
        responses_df.to_csv(responses_path, index=False)
        logger.info(f"Responses saved to {responses_path}")
    else:
        logger.info("Skipping generation, loading existing responses...")
        import pandas as pd
        responses_df = pd.read_csv(output_dir / 'responses.csv')

    # Step 4: Evaluate Safety
    logger.info("=" * 50)
    logger.info("Step 4: Evaluating safety...")
    logger.info("=" * 50)
    results_df = evaluate_safety(
        responses_df=responses_df,
        detoxify_model=eval_config['evaluation']['detoxify_model']
    )

    # Save evaluation results
    results_path = output_dir / 'evaluation_results.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"Evaluation results saved to {results_path}")

    # Print summary statistics
    logger.info("=" * 50)
    logger.info("Summary Statistics")
    logger.info("=" * 50)
    toxicity_cols = eval_config['evaluation']['metrics']
    summary = results_df.groupby('category')[toxicity_cols].mean()
    logger.info(f"\n{summary}")

    # Step 5: Create Visualization
    logger.info("=" * 50)
    logger.info("Step 5: Creating dashboard...")
    logger.info("=" * 50)
    dashboard_path = output_dir / eval_config['output']['dashboard_filename']
    create_dashboard(
        evaluated_df=results_df,
        output_path=str(dashboard_path)
    )
    logger.info(f"Dashboard saved to {dashboard_path}")

    # Final summary
    logger.info("=" * 50)
    logger.info("Evaluation Complete!")
    logger.info("=" * 50)
    logger.info(f"Total prompts evaluated: {len(results_df)}")
    logger.info(f"Results directory: {output_dir}")
    logger.info(f"Key outputs:")
    logger.info(f"  - Responses: {results_path}")
    logger.info(f"  - Dashboard: {dashboard_path}")
    logger.info(f"  - Evaluation results: {results_path}")


if __name__ == "__main__":
    main()
