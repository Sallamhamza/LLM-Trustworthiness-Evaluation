"""
Model Setup and Initialization Module

This module handles loading and configuring Large Language Models
for safety evaluation. Supports GPU optimization and memory management.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles model and tokenizer loading with GPU optimization."""

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize ModelLoader.

        Args:
            model_name: Hugging Face model identifier
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def load_model(self,
                   max_memory: Optional[dict] = None,
                   torch_dtype: torch.dtype = torch.float16) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer with optimal configuration.

        Args:
            max_memory: Memory allocation per device
            torch_dtype: Data type for model weights

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {self.model_name}")

        # Clear GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            max_memory=max_memory,
            trust_remote_code=True
        )

        if self.device == "cpu":
            model = model.to(self.device)

        model.eval()  # Set to evaluation mode

        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

        return model, tokenizer

    def clear_memory(self):
        """Clear GPU memory."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU memory cleared")


def load_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
               device: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Convenience function to load a model.

    Args:
        model_name: Hugging Face model identifier
        device: Target device ('cuda' or 'cpu')

    Returns:
        Tuple of (model, tokenizer)
    """
    loader = ModelLoader(model_name)
    return loader.load_model()


def get_device_info() -> dict:
    """
    Get information about available compute devices.

    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info


if __name__ == "__main__":
    # Example usage
    device_info = get_device_info()
    print("Device Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")

    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded successfully!")
