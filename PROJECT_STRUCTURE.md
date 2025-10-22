# Project Structure

This document provides a complete overview of the repository structure.

## Directory Tree

```
llm-safety-evaluation/
│
├── README.md                          # Main project documentation
├── LICENSE                            # MIT License
├── CONTRIBUTING.md                    # Contribution guidelines
├── PROJECT_STRUCTURE.md              # This file
├── requirements.txt                   # Python dependencies
├── main.py                           # Main entry point script
├── .gitignore                        # Git ignore rules
│
├── config/                           # Configuration files
│   ├── model_config.json            # Model settings (temperature, max_length, etc.)
│   └── evaluation_config.json       # Evaluation parameters (samples, metrics, etc.)
│
├── src/                              # Source code modules
│   ├── __init__.py                  # Package initialization
│   ├── model_setup.py               # Model loading and GPU optimization
│   ├── data_loader.py               # Dataset loading and sampling
│   ├── generator.py                 # Response generation with checkpointing
│   ├── evaluator.py                 # Safety evaluation (toxicity, bias)
│   ├── visualization.py             # Dashboard and plotting
│   └── defenses/                    # Defense mechanisms
│       ├── __init__.py
│       ├── privacy_protection.py    # PII detection and filtering
│       └── stereotype_mitigation.py # Bias reduction techniques
│
├── notebooks/                        # Jupyter notebooks
│   └── complete_evaluation_pipeline.ipynb  # Full evaluation notebook
│
├── data/                            # Data directory (gitignored)
│   ├── checkpoints/                 # Generation checkpoints
│   └── results/                     # Evaluation results and visualizations
│
├── docs/                            # Documentation
│   ├── SETUP.md                     # Installation and setup guide
│   └── USAGE.md                     # Detailed usage instructions
│
└── tests/                           # Unit tests (to be added)
    └── test_evaluation.py
```

## File Descriptions

### Root Level Files

| File | Purpose |
|------|---------|
| `README.md` | Main project documentation, features, quick start |
| `LICENSE` | MIT License for the project |
| `CONTRIBUTING.md` | Guidelines for contributors |
| `requirements.txt` | Python package dependencies |
| `main.py` | Command-line interface for running evaluations |
| `.gitignore` | Files and directories to exclude from git |

### Configuration (`config/`)

| File | Purpose |
|------|---------|
| `model_config.json` | Model name, generation parameters, memory settings |
| `evaluation_config.json` | Dataset config, evaluation metrics, output settings |

### Source Code (`src/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `model_setup.py` | Model loading and initialization | `load_model()`, `get_device_info()` |
| `data_loader.py` | Dataset loading and sampling | `load_benchmark_datasets()` |
| `generator.py` | Response generation pipeline | `generate_responses()` |
| `evaluator.py` | Safety metrics and scoring | `evaluate_safety()` |
| `visualization.py` | Dashboard creation | `create_dashboard()` |

### Defense Mechanisms (`src/defenses/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `privacy_protection.py` | PII detection and redaction | `PrivacyProtectionSystem` |
| `stereotype_mitigation.py` | Bias detection and mitigation | `StereotypeMitigator` |

### Documentation (`docs/`)

| File | Purpose |
|------|---------|
| `SETUP.md` | Installation instructions, troubleshooting |
| `USAGE.md` | Detailed usage examples, API reference |

## Module Dependencies

```
main.py
  └── src/
      ├── model_setup.py
      │   └── torch, transformers
      ├── data_loader.py
      │   └── datasets, pandas
      ├── generator.py
      │   ├── model_setup.py
      │   └── torch, transformers
      ├── evaluator.py
      │   └── detoxify, pandas
      ├── visualization.py
      │   └── matplotlib, seaborn
      └── defenses/
          ├── privacy_protection.py
          └── stereotype_mitigation.py
```

## Data Flow

```
1. Configuration Loading
   ├── config/model_config.json
   └── config/evaluation_config.json
          ↓
2. Model Setup
   └── src/model_setup.py → Load TinyLlama model
          ↓
3. Data Loading
   └── src/data_loader.py → Load DecodingTrust datasets
          ↓
4. Response Generation
   └── src/generator.py → Generate 600 responses
          ↓
5. Safety Evaluation
   └── src/evaluator.py → Compute toxicity scores
          ↓
6. Visualization
   └── src/visualization.py → Create dashboard
          ↓
7. Results Output
   └── data/results/ → Save CSV, PNG files
```

## Usage Patterns

### Pattern 1: Using main.py (Recommended)

```bash
python main.py --num-samples 600 --output-dir data/results
```

### Pattern 2: Using Python API

```python
from src import load_model, load_benchmark_datasets, generate_responses
from src import evaluate_safety, create_dashboard

model, tokenizer = load_model()
prompts = load_benchmark_datasets(600)
responses = generate_responses(model, tokenizer, prompts)
results = evaluate_safety(responses)
create_dashboard(results)
```

### Pattern 3: Using Jupyter Notebook

Open `notebooks/complete_evaluation_pipeline.ipynb` in Jupyter or Google Colab.

## Configuration Options

### Model Configuration (`config/model_config.json`)

```json
{
  "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "generation": {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9
  }
}
```

### Evaluation Configuration (`config/evaluation_config.json`)

```json
{
  "dataset": {
    "total_samples": 600,
    "categories": ["toxicity", "stereotype", "privacy"]
  },
  "evaluation": {
    "detoxify_model": "original",
    "toxicity_threshold": 0.7
  }
}
```

## Output Files

The framework generates the following outputs in `data/results/`:

| File | Description |
|------|-------------|
| `responses.csv` | Generated responses for all prompts |
| `evaluation_results.csv` | Complete evaluation with toxicity scores |
| `evaluation_dashboard.png` | 9-subplot visualization dashboard |
| `summary_statistics.csv` | Summary statistics per category |
| `high_risk_examples.csv` | Examples with high toxicity scores |

## Extensibility

### Adding New Evaluation Metrics

1. Extend `SafetyEvaluator` class in `src/evaluator.py`
2. Add new metric computation method
3. Update `evaluation_config.json` with new metric
4. Extend visualization in `src/visualization.py`

### Adding New Defense Mechanisms

1. Create new file in `src/defenses/`
2. Implement defense class with standard interface
3. Update `src/defenses/__init__.py`
4. Add configuration in `evaluation_config.json`

### Adding New Datasets

1. Extend `BenchmarkDataLoader` in `src/data_loader.py`
2. Add dataset loading method
3. Update configuration to include new dataset

## Version Information

- **Current Version**: 1.0.0
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Transformers**: 4.35+

## Support

For questions or issues:
- Check documentation in `docs/`
- Review examples in `notebooks/`
- Open an issue on GitHub
- Contact maintainers

## License

MIT License - See [LICENSE](LICENSE) file for details.
