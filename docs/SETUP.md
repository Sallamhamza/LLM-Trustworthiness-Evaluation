# Setup Guide

This guide will help you set up the LLM Safety Evaluation Framework on your local machine or cloud environment.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (T4, V100, A100, or better)
  - Minimum 16GB VRAM recommended
  - CPU-only mode is supported but significantly slower
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: At least 10GB free space for models and data

### Software Requirements
- **Operating System**: Linux, macOS, or Windows (with WSL2 recommended)
- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (for GPU support)
- **Git**: For version control

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/llm-safety-evaluation.git
cd llm-safety-evaluation
```

### 2. Create Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n llm-safety python=3.10
conda activate llm-safety
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For development (includes testing tools):
```bash
pip install -r requirements.txt
pip install pytest pytest-cov black flake8
```

### 4. Install Spacy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Set Up Hugging Face Authentication

You'll need a Hugging Face account and token to download models and datasets.

1. Create an account at [huggingface.co](https://huggingface.co)
2. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Set the token as an environment variable:

**Linux/macOS:**
```bash
export HF_TOKEN="your_token_here"
```

**Windows:**
```bash
set HF_TOKEN=your_token_here
```

**Or create a .env file:**
```bash
echo "HF_TOKEN=your_token_here" > .env
```

### 6. Verify CUDA Installation (GPU users)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Expected output:
```
CUDA available: True
CUDA version: 11.8
```

## Configuration

### Model Configuration

Edit `config/model_config.json` to customize model settings:

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

### Evaluation Configuration

Edit `config/evaluation_config.json` to customize evaluation settings:

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

## Quick Test

Run a quick test to verify installation:

```python
from src.model_setup import load_model, get_device_info

# Check device
print(get_device_info())

# Load model
model, tokenizer = load_model()
print("Setup successful!")
```

## Google Colab Setup

If using Google Colab:

1. Upload the notebook to Google Drive
2. Open with Google Colab
3. Enable GPU:
   - Runtime → Change runtime type → GPU (T4)
4. Install dependencies in a notebook cell:

```python
!pip install -q transformers datasets accelerate detoxify spacy
!python -m spacy download en_core_web_sm
```

5. Mount Google Drive for data persistence:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors:

1. Reduce batch size in `config/evaluation_config.json`:
```json
{
  "generation": {
    "batch_size": 25
  }
}
```

2. Use smaller model:
```json
{
  "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}
```

3. Enable CPU offloading in model config

### Slow Installation

If pip installation is slow:

```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Hugging Face Authentication Issues

If datasets fail to load:

```bash
huggingface-cli login
# Enter your token when prompted
```

### Spacy Model Not Found

```bash
python -m spacy download en_core_web_sm --force
```

## Next Steps

After successful setup:

1. Read the [Usage Guide](USAGE.md) for detailed instructions
2. Review the [Methodology](METHODOLOGY.md) to understand the evaluation process
3. Check out example notebooks in the `notebooks/` directory
4. Run your first evaluation!

## Support

For additional help:
- Check [GitHub Issues](https://github.com/yourusername/llm-safety-evaluation/issues)
- Review the main [README.md](../README.md)
- Contact the maintainers

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8GB | 16GB+ |
| System RAM | 16GB | 32GB+ |
| Storage | 5GB | 10GB+ |
| Python | 3.8 | 3.10+ |
| CUDA | 11.0 | 11.8+ |
