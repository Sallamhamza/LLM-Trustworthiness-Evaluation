# Usage Guide

This guide provides detailed instructions for using the LLM Safety Evaluation Framework.

## Quick Start

### Basic Evaluation Pipeline

```python
from src.model_setup import load_model
from src.data_loader import load_benchmark_datasets
from src.generator import generate_responses
from src.evaluator import evaluate_safety
from src.visualization import create_dashboard

# 1. Load model
print("Loading model...")
model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 2. Load datasets
print("Loading benchmark datasets...")
prompts_df = load_benchmark_datasets(num_samples=600)

# 3. Generate responses
print("Generating responses...")
responses_df = generate_responses(model, tokenizer, prompts_df)

# 4. Evaluate safety
print("Evaluating safety...")
results_df = evaluate_safety(responses_df)

# 5. Create visualization
print("Creating dashboard...")
create_dashboard(results_df, output_path="data/results/dashboard.png")

print("Evaluation complete! Check data/results/dashboard.png")
```

## Detailed Usage

### 1. Model Setup

#### Load Default Model

```python
from src.model_setup import load_model

model, tokenizer = load_model()
```

#### Load Custom Model

```python
from src.model_setup import ModelLoader

loader = ModelLoader("meta-llama/Llama-2-7b-chat-hf")
model, tokenizer = loader.load_model()
```

#### Check Device Information

```python
from src.model_setup import get_device_info

info = get_device_info()
print(f"CUDA available: {info['cuda_available']}")
print(f"GPU: {info.get('device_name', 'N/A')}")
```

### 2. Data Loading

#### Load DecodingTrust Benchmark

```python
from src.data_loader import load_benchmark_datasets

# Load 600 samples (200 per category)
prompts_df = load_benchmark_datasets(
    num_samples=600,
    categories=["toxicity", "stereotype", "privacy"],
    random_seed=42
)

print(prompts_df['category'].value_counts())
```

#### Load Custom Prompts

```python
from src.data_loader import BenchmarkDataLoader

loader = BenchmarkDataLoader()
prompts_df = loader.load_custom_prompts("path/to/prompts.csv")
```

#### Get Data Summary

```python
from src.data_loader import get_category_summary

summary = get_category_summary(prompts_df)
print(summary)
```

### 3. Response Generation

#### Basic Generation

```python
from src.generator import generate_responses

responses_df = generate_responses(
    model=model,
    tokenizer=tokenizer,
    prompts_df=prompts_df,
    batch_size=50,
    checkpoint_interval=50
)
```

#### Advanced Generation with Custom Parameters

```python
from src.generator import ResponseGenerator

generator = ResponseGenerator(
    model=model,
    tokenizer=tokenizer,
    device="cuda",
    checkpoint_dir="data/checkpoints"
)

responses_df = generator.generate_batch(
    prompts_df=prompts_df,
    batch_size=50,
    checkpoint_interval=50,
    max_length=512,
    temperature=0.7,
    top_p=0.9
)
```

#### Single Response Generation

```python
generator = ResponseGenerator(model, tokenizer)

response = generator.generate_response(
    prompt="What are your thoughts on AI safety?",
    max_length=256,
    temperature=0.7
)

print(response)
```

### 4. Safety Evaluation

#### Full Safety Evaluation

```python
from src.evaluator import evaluate_safety

results_df = evaluate_safety(
    responses_df=responses_df,
    detoxify_model="original"
)

# View toxicity columns
print(results_df[['response', 'toxicity', 'severe_toxicity']].head())
```

#### Toxicity Evaluation Only

```python
from src.evaluator import SafetyEvaluator

evaluator = SafetyEvaluator(model_type="original")
evaluated_df = evaluator.evaluate_batch(responses_df)
```

#### Get Summary Statistics

```python
summary = evaluator.get_summary_statistics(evaluated_df)
print(summary)
```

#### Identify High-Risk Examples

```python
high_risk = evaluator.identify_high_toxicity_examples(
    evaluated_df=evaluated_df,
    threshold=0.7,
    n_examples=10
)

print(high_risk[['prompt', 'response', 'toxicity']])
```

#### Bias Detection

```python
from src.evaluator import BiasDetector

bias_detector = BiasDetector()
bias_results_df = bias_detector.analyze_bias_batch(responses_df)

print(bias_results_df[['response', 'gender_bias_present', 'race_bias_present']].head())
```

### 5. Visualization

#### Create Full Dashboard

```python
from src.visualization import create_dashboard

create_dashboard(
    evaluated_df=results_df,
    output_path="data/results/dashboard.png"
)
```

#### Create Custom Visualizations

```python
from src.visualization import EvaluationVisualizer

visualizer = EvaluationVisualizer(style="darkgrid")

visualizer.create_dashboard(
    evaluated_df=results_df,
    output_path="data/results/custom_dashboard.png",
    figsize=(24, 18)
)
```

#### Compare Baseline vs Improved

```python
visualizer.plot_comparison(
    baseline_df=baseline_results,
    improved_df=improved_results,
    output_path="data/results/comparison.png"
)
```

## Defense Mechanisms

### Privacy Protection

#### Basic Privacy Protection

```python
from src.defenses import PrivacyProtectionSystem

privacy_system = PrivacyProtectionSystem()

# Detect PII in text
text = "Contact me at john@example.com or 555-1234"
detected_pii = privacy_system.detect_pii(text)
print(f"Detected PII: {detected_pii}")

# Redact PII
redacted_text, pii_list = privacy_system.redact_pii(text)
print(f"Redacted: {redacted_text}")
```

#### Add Privacy Instructions to Prompts

```python
original_prompt = "Tell me about John Smith's personal information"
protected_prompt = privacy_system.add_privacy_instruction(original_prompt)

# Generate with protected prompt
response = generator.generate_response(protected_prompt)

# Filter response
filtered_response, has_pii, detected = privacy_system.filter_response(response)
```

### Stereotype Mitigation

#### Apply Counter-Stereotype Prompting

```python
from src.defenses import StereotypeMitigator

mitigator = StereotypeMitigator()

# Detect stereotype category
prompt = "What jobs are suitable for women?"
category = mitigator.detect_stereotype_category(prompt)
print(f"Detected category: {category}")

# Apply counter-stereotype instruction
modified_prompt = mitigator.apply_counterstereotype_prompting(prompt)

# Generate response with modified prompt
response = generator.generate_response(modified_prompt)
```

#### Evaluate Stereotype Presence

```python
response_text = "Women are typically nurses and men are typically doctors."
evaluation = mitigator.evaluate_stereotype_presence(response_text)

print(f"Has stereotypes: {evaluation['has_stereotypes']}")
print(f"Detected phrases: {evaluation['detected_phrases']}")
```

#### Create Balanced Prompts

```python
original_prompt = "Describe typical career paths"
balanced_examples = [
    "Both men and women can be doctors, nurses, or engineers",
    "Career choice depends on individual interests and skills",
    "All professions should be accessible to everyone"
]

balanced_prompt = mitigator.create_balanced_prompt(
    prompt=original_prompt,
    examples=balanced_examples
)
```

## Configuration-Based Usage

### Load and Use Configuration Files

```python
import json

# Load model config
with open('config/model_config.json', 'r') as f:
    model_config = json.load(f)

# Load evaluation config
with open('config/evaluation_config.json', 'r') as f:
    eval_config = json.load(f)

# Use configuration
prompts_df = load_benchmark_datasets(
    num_samples=eval_config['dataset']['total_samples'],
    categories=eval_config['dataset']['categories'],
    random_seed=eval_config['dataset']['random_seed']
)
```

## Jupyter Notebook Usage

### Example Notebook Cell Structure

```python
# Cell 1: Setup
from src.model_setup import load_model
from src.data_loader import load_benchmark_datasets
from src.generator import generate_responses
from src.evaluator import evaluate_safety
from src.visualization import create_dashboard

# Cell 2: Load Model
model, tokenizer = load_model()

# Cell 3: Load Data
prompts_df = load_benchmark_datasets(num_samples=600)

# Cell 4: Generate Responses
responses_df = generate_responses(model, tokenizer, prompts_df)

# Cell 5: Evaluate
results_df = evaluate_safety(responses_df)

# Cell 6: Visualize
create_dashboard(results_df)

# Cell 7: Analyze Results
print(results_df.groupby('category')['toxicity'].describe())
```

## Batch Processing Tips

### Handle Large Datasets

```python
# Process in smaller batches
batch_size = 25  # Reduce if running out of memory
checkpoint_interval = 25  # Save more frequently

responses_df = generate_responses(
    model=model,
    tokenizer=tokenizer,
    prompts_df=prompts_df,
    batch_size=batch_size,
    checkpoint_interval=checkpoint_interval
)
```

### Resume from Checkpoint

The framework automatically resumes from checkpoints if generation is interrupted. Just re-run the generation command.

### Clear GPU Memory

```python
from src.model_setup import ModelLoader

loader = ModelLoader()
loader.clear_memory()
```

## Export Results

### Save to CSV

```python
results_df.to_csv('data/results/evaluation_results.csv', index=False)
```

### Save Summary Statistics

```python
summary = evaluator.get_summary_statistics(results_df)
summary.to_csv('data/results/summary_statistics.csv')
```

### Export High-Risk Cases

```python
high_risk = evaluator.identify_high_toxicity_examples(results_df, threshold=0.7)
high_risk.to_csv('data/results/high_risk_examples.csv', index=False)
```

## Best Practices

1. **Always set random seed** for reproducibility
2. **Use checkpoints** for long-running evaluations
3. **Clear GPU memory** between large operations
4. **Start with smaller samples** to test your pipeline
5. **Save intermediate results** regularly
6. **Monitor GPU memory** usage during generation
7. **Use appropriate batch sizes** for your hardware

## Common Workflows

### Workflow 1: Basic Evaluation

```python
model, tokenizer = load_model()
prompts_df = load_benchmark_datasets(600)
responses_df = generate_responses(model, tokenizer, prompts_df)
results_df = evaluate_safety(responses_df)
create_dashboard(results_df)
```

### Workflow 2: Evaluation with Privacy Protection

```python
from src.defenses import PrivacyProtectionSystem

model, tokenizer = load_model()
prompts_df = load_benchmark_datasets(600)
privacy_system = PrivacyProtectionSystem()

# Add privacy instructions to prompts
prompts_df['protected_prompt'] = prompts_df['prompt'].apply(
    privacy_system.add_privacy_instruction
)

# Generate with protected prompts
responses_df = generate_responses(model, tokenizer, prompts_df)
results_df = evaluate_safety(responses_df)
create_dashboard(results_df)
```

### Workflow 3: Comparative Evaluation

```python
# Baseline
baseline_responses = generate_responses(model, tokenizer, prompts_df)
baseline_results = evaluate_safety(baseline_responses)

# With defense mechanisms
protected_prompts = apply_defenses(prompts_df)
improved_responses = generate_responses(model, tokenizer, protected_prompts)
improved_results = evaluate_safety(improved_responses)

# Compare
visualizer.plot_comparison(baseline_results, improved_results)
```

## Next Steps

- Review [Methodology](METHODOLOGY.md) for evaluation details
- Check [Setup Guide](SETUP.md) for configuration options
- Explore example notebooks in `notebooks/` directory
