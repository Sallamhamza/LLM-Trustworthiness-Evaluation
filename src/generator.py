"""
Response Generation Module

Handles batch response generation from LLMs with checkpoint recovery
and memory optimization.
"""

import torch
import pandas as pd
from typing import List, Dict, Optional
import logging
import gc
import json
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates responses from LLMs with checkpointing."""

    def __init__(self,
                 model,
                 tokenizer,
                 device: str = "cuda",
                 checkpoint_dir: str = "data/checkpoints"):
        """
        Initialize response generator.

        Args:
            model: Loaded language model
            tokenizer: Model tokenizer
            device: Compute device
            checkpoint_dir: Directory for checkpoint files
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def generate_response(self,
                         prompt: str,
                         max_length: int = 512,
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> str:
        """
        Generate a single response for a prompt.

        Args:
            prompt: Input prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated response text
        """
        # Format prompt for chat models
        formatted_prompt = self._format_chat_prompt(prompt)

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        response = self._extract_assistant_response(response, prompt)

        return response

    def _format_chat_prompt(self, prompt: str) -> str:
        """
        Format prompt for chat models (e.g., TinyLlama-Chat).

        Args:
            prompt: Raw prompt text

        Returns:
            Formatted prompt with chat template
        """
        # Check if model uses special chat tokens
        if "<|user|>" in self.tokenizer.special_tokens_map.get("additional_special_tokens", []):
            return f"<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            return prompt

    def _extract_assistant_response(self, full_text: str, original_prompt: str) -> str:
        """
        Extract assistant response from generated text.

        Args:
            full_text: Full generated text
            original_prompt: Original input prompt

        Returns:
            Extracted assistant response
        """
        # Try to extract response after assistant token
        if "<|assistant|>" in full_text:
            parts = full_text.split("<|assistant|>")
            if len(parts) > 1:
                return parts[-1].strip()

        # Fallback: remove the prompt from the beginning
        if full_text.startswith(original_prompt):
            return full_text[len(original_prompt):].strip()

        return full_text.strip()

    def generate_batch(self,
                      prompts_df: pd.DataFrame,
                      batch_size: int = 50,
                      checkpoint_interval: int = 50,
                      **generation_kwargs) -> pd.DataFrame:
        """
        Generate responses for a batch of prompts with checkpointing.

        Args:
            prompts_df: DataFrame with 'prompt' column
            batch_size: Number of prompts per batch
            checkpoint_interval: Save checkpoint every N prompts
            **generation_kwargs: Additional arguments for generation

        Returns:
            DataFrame with prompts and responses
        """
        results = []
        checkpoint_file = self.checkpoint_dir / "generation_checkpoint.json"

        # Load existing checkpoint if available
        start_idx = 0
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data['results']
                start_idx = checkpoint_data['last_index']
                logger.info(f"Resuming from checkpoint at index {start_idx}")

        # Generate responses
        for idx in tqdm(range(start_idx, len(prompts_df)), desc="Generating responses"):
            row = prompts_df.iloc[idx]
            prompt = row['prompt']

            try:
                response = self.generate_response(prompt, **generation_kwargs)

                result = {
                    'prompt': prompt,
                    'response': response,
                    'category': row.get('category', 'unknown'),
                    'split': row.get('split', 'unknown')
                }
                results.append(result)

                # Checkpoint every N samples
                if (idx + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(results, idx + 1, checkpoint_file)
                    self._clear_memory()

            except Exception as e:
                logger.error(f"Error generating response for prompt {idx}: {e}")
                results.append({
                    'prompt': prompt,
                    'response': f"ERROR: {str(e)}",
                    'category': row.get('category', 'unknown'),
                    'split': row.get('split', 'unknown')
                })

        # Final save
        self._save_checkpoint(results, len(prompts_df), checkpoint_file)

        logger.info(f"Generated {len(results)} responses")
        return pd.DataFrame(results)

    def _save_checkpoint(self, results: List[Dict], last_index: int, checkpoint_file: Path):
        """Save checkpoint to file."""
        checkpoint_data = {
            'results': results,
            'last_index': last_index
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        logger.info(f"Checkpoint saved at index {last_index}")

    def _clear_memory(self):
        """Clear GPU memory."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()


def generate_responses(model,
                      tokenizer,
                      prompts_df: pd.DataFrame,
                      batch_size: int = 50,
                      checkpoint_interval: int = 50,
                      device: str = None) -> pd.DataFrame:
    """
    Convenience function to generate responses.

    Args:
        model: Loaded language model
        tokenizer: Model tokenizer
        prompts_df: DataFrame with prompts
        batch_size: Batch size for generation
        checkpoint_interval: Checkpoint frequency
        device: Compute device

    Returns:
        DataFrame with responses
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = ResponseGenerator(model, tokenizer, device=device)
    return generator.generate_batch(
        prompts_df,
        batch_size=batch_size,
        checkpoint_interval=checkpoint_interval
    )


if __name__ == "__main__":
    print("Response Generator Module")
    print("Use this module to generate LLM responses for evaluation.")
