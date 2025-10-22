"""
Defense Mechanisms Package

Implements various safety improvement techniques for LLMs.
"""

from .privacy_protection import PrivacyProtectionSystem
from .stereotype_mitigation import StereotypeMitigator

__all__ = ['PrivacyProtectionSystem', 'StereotypeMitigator']
