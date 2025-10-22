"""
Privacy Protection Module

Implements multi-layer privacy protection system to prevent
personally identifiable information (PII) leakage.
"""

import re
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrivacyProtectionSystem:
    """Multi-layer system for protecting privacy in LLM responses."""

    def __init__(self):
        """Initialize privacy protection system."""
        self.pii_patterns = self._initialize_pii_patterns()

    def _initialize_pii_patterns(self) -> Dict[str, re.Pattern]:
        """
        Initialize regex patterns for PII detection.

        Returns:
            Dictionary of PII type to regex pattern
        """
        patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            'url': re.compile(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'),
        }
        return patterns

    def detect_pii(self, text: str) -> List[Dict[str, str]]:
        """
        Detect PII in text.

        Args:
            text: Input text to scan

        Returns:
            List of detected PII with type and value
        """
        detected_pii = []

        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                detected_pii.append({
                    'type': pii_type,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })

        return detected_pii

    def redact_pii(self, text: str, replacement: str = "[REDACTED]") -> Tuple[str, List[Dict]]:
        """
        Redact PII from text.

        Args:
            text: Input text
            replacement: Replacement string for PII

        Returns:
            Tuple of (redacted text, list of detected PII)
        """
        detected_pii = self.detect_pii(text)
        redacted_text = text

        # Sort by start position in reverse to avoid offset issues
        for pii in sorted(detected_pii, key=lambda x: x['start'], reverse=True):
            redacted_text = (
                redacted_text[:pii['start']] +
                f"{replacement}_{pii['type'].upper()}" +
                redacted_text[pii['end']:]
            )

        return redacted_text, detected_pii

    def add_privacy_instruction(self, prompt: str) -> str:
        """
        Add privacy protection instruction to prompt.

        Args:
            prompt: Original prompt

        Returns:
            Modified prompt with privacy instruction
        """
        privacy_instruction = (
            "IMPORTANT: Do not include any personally identifiable information (PII) "
            "such as names, addresses, phone numbers, email addresses, social security numbers, "
            "or any other sensitive personal data in your response.\n\n"
        )

        return privacy_instruction + prompt

    def filter_response(self, response: str) -> Tuple[str, bool, List[Dict]]:
        """
        Filter response for privacy violations.

        Args:
            response: LLM response to filter

        Returns:
            Tuple of (filtered response, has_pii, detected_pii)
        """
        detected_pii = self.detect_pii(response)
        has_pii = len(detected_pii) > 0

        if has_pii:
            filtered_response, _ = self.redact_pii(response)
            logger.warning(f"PII detected and redacted: {len(detected_pii)} instances")
            return filtered_response, has_pii, detected_pii
        else:
            return response, has_pii, []

    def apply_privacy_protection(self,
                                prompt: str,
                                model,
                                tokenizer,
                                **generation_kwargs) -> Dict:
        """
        Apply privacy protection to generation pipeline.

        Args:
            prompt: Input prompt
            model: Language model
            tokenizer: Tokenizer
            **generation_kwargs: Generation parameters

        Returns:
            Dictionary with protected response and metadata
        """
        # Add privacy instruction
        protected_prompt = self.add_privacy_instruction(prompt)

        # Generate response (simplified - integrate with actual generator)
        # This is a placeholder for the actual generation logic
        response = "Generated response here"  # Replace with actual generation

        # Filter response
        filtered_response, has_pii, detected_pii = self.filter_response(response)

        return {
            'original_prompt': prompt,
            'protected_prompt': protected_prompt,
            'response': filtered_response,
            'has_pii': has_pii,
            'detected_pii_count': len(detected_pii),
            'detected_pii': detected_pii
        }


class PIIDetector:
    """Standalone PII detection utility."""

    def __init__(self):
        """Initialize PII detector."""
        self.protection_system = PrivacyProtectionSystem()

    def scan_text(self, text: str) -> Dict:
        """
        Scan text for PII.

        Args:
            text: Input text

        Returns:
            Dictionary with scan results
        """
        detected_pii = self.protection_system.detect_pii(text)

        return {
            'text': text,
            'has_pii': len(detected_pii) > 0,
            'pii_count': len(detected_pii),
            'pii_types': list(set(pii['type'] for pii in detected_pii)),
            'detected_pii': detected_pii
        }


if __name__ == "__main__":
    print("Privacy Protection Module")

    # Example usage
    protection_system = PrivacyProtectionSystem()

    test_text = "Contact me at john.doe@example.com or call 555-123-4567"
    print(f"\nOriginal text: {test_text}")

    detected = protection_system.detect_pii(test_text)
    print(f"Detected PII: {detected}")

    redacted, pii_list = protection_system.redact_pii(test_text)
    print(f"Redacted text: {redacted}")
