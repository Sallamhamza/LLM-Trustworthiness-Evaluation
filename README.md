# LLM Safety Evaluation Framework

A comprehensive framework for evaluating the safety, trustworthiness, and robustness of Large Language Models (LLMs) using standardized benchmark datasets and multi-dimensional evaluation metrics.

## Overview

This project provides a systematic approach to assess LLM behavior across critical safety dimensions including toxicity, bias, stereotypes, and privacy vulnerabilities. The framework uses adversarial prompts from established benchmarks to generate comprehensive safety reports and supports the implementation of defense mechanisms.

## Key Features

- **Multi-Dimensional Safety Evaluation**: Assess models across toxicity, bias, and privacy dimensions
- **Benchmark Dataset Integration**: Built-in support for DecodingTrust and TrustLLM datasets
- **Automated Evaluation Pipeline**: Batch processing with checkpoint recovery for large-scale testing
- **Comprehensive Visualization**: Generate statistical dashboards with 9-subplot analysis
- **Defense Mechanisms**: Extensible architecture for implementing and testing safety improvements
- **GPU-Optimized**: Memory-efficient processing with CUDA acceleration support

## Architecture

```
├── config/                 # Configuration files for models and evaluation
├── src/                    # Core source code modules
│   ├── model_setup.py     # Model loading and initialization
│   ├── data_loader.py     # Dataset handling and sampling
│   ├── generator.py       # Response generation pipeline
│   ├── evaluator.py       # Safety metrics and scoring
│   ├── visualization.py   # Dashboard and plotting utilities
│   └── defenses/          # Safety improvement implementations
├── notebooks/             # Jupyter notebooks for experimentation
├── data/                  # Data directory (excluded from git)
│   └── results/          # Generated evaluation results
├── docs/                  # Additional documentation
└── tests/                 # Unit tests

```

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (T4 or better recommended)
- 16GB+ RAM
- 5GB+ storage for model downloads

### Software
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- PyTorch 2.0+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-safety-evaluation.git
cd llm-safety-evaluation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Hugging Face authentication:
```bash
# Set your Hugging Face token as an environment variable
export HF_TOKEN="your_token_here"
```

## Quick Start

### Using the Complete Notebook

For a guided walkthrough, open the comprehensive notebook:
```bash
jupyter notebook notebooks/complete_evaluation_pipeline.ipynb
```

### Using Python Scripts

```python
from src.model_setup import load_model
from src.data_loader import load_benchmark_datasets
from src.generator import generate_responses
from src.evaluator import evaluate_safety
from src.visualization import create_dashboard

# Load model
model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Load benchmark datasets
prompts = load_benchmark_datasets(num_samples=600)

# Generate responses
responses = generate_responses(model, tokenizer, prompts)

# Evaluate safety
results = evaluate_safety(responses)

# Create visualization
create_dashboard(results, output_path="results/dashboard.png")
```

## Evaluation Metrics

The framework evaluates models across multiple dimensions:

| Dimension | Metrics | Description |
|-----------|---------|-------------|
| **Toxicity** | Detoxify scores | Measures harmful, offensive, or toxic content generation |
| **Bias** | Stereotype analysis | Identifies gender, race, and occupational biases |
| **Privacy** | PII leakage detection | Detects privacy violations and personal information exposure |

## Benchmark Datasets

### DecodingTrust
- **Toxicity subset**: Multiple prompt variants designed to elicit toxic responses
- **Stereotype subset**: Tests for gender, race, and occupation biases
- **Privacy subset**: Prompts designed to test privacy protection

### TrustLLM
- Complementary safety-focused benchmark dataset
- Aligned with DecodingTrust categories

## Configuration

Customize model and evaluation settings in `config/`:

**model_config.json**
```json
{
  "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "max_length": 512,
  "temperature": 0.7,
  "device": "cuda"
}
```

**evaluation_config.json**
```json
{
  "num_samples": 600,
  "batch_size": 50,
  "checkpoint_interval": 50,
  "random_seed": 42
}
```

## Defense Mechanisms (Optional)

Implement and test safety improvements:

```python
from src.defenses.privacy_protection import PrivacyProtectionSystem
from src.defenses.stereotype_mitigation import StereotypeMitigator

# Apply privacy protection
privacy_system = PrivacyProtectionSystem(model, tokenizer)
safe_responses = privacy_system.generate_safe_responses(prompts)

# Apply stereotype mitigation
mitigator = StereotypeMitigator()
improved_responses = mitigator.apply_counterstereotype_prompting(prompts)
```

## Results

The evaluation pipeline generates:
- **Detailed metrics**: Statistical analysis per safety dimension
- **Visual dashboards**: 9-subplot comprehensive visualization
- **Example documentation**: Specific cases illustrating vulnerabilities
- **Comparison reports**: Baseline vs. improved performance (if defense mechanisms applied)

## Project Structure Details

### Core Modules

- **model_setup.py**: Handles model loading, tokenizer configuration, and GPU optimization
- **data_loader.py**: Manages dataset loading, stratified sampling, and prompt preprocessing
- **generator.py**: Implements batch response generation with checkpoint recovery
- **evaluator.py**: Provides toxicity scoring, bias detection, and privacy analysis
- **visualization.py**: Creates statistical plots and comprehensive dashboards

### Defense Implementations

- **privacy_protection.py**: Multi-layer PII detection and filtering system
- **stereotype_mitigation.py**: Counter-stereotype prompting and bias reduction

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{llm_safety_evaluation,
  title={LLM Safety Evaluation Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/llm-safety-evaluation}
}
```

## Acknowledgments

- Built on the [DecodingTrust](https://github.com/AI-secure/DecodingTrust) benchmark
- Uses the [Detoxify](https://github.com/unitaryai/detoxify) library for toxicity scoring
- Powered by [Hugging Face Transformers](https://huggingface.co/transformers)

## References

- Wang et al. (2023). "DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models"
- TrustLLM Benchmark Documentation
- TinyLlama Model Card

## Support

For questions or issues, please open an issue on GitHub or contact the maintainers.
