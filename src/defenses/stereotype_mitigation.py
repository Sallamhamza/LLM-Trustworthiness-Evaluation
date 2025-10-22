"""
Stereotype Mitigation Module

Implements counter-stereotype prompting and bias reduction techniques.
"""

from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StereotypeMitigator:
    """Implements stereotype mitigation strategies."""

    def __init__(self):
        """Initialize stereotype mitigator."""
        self.counter_stereotype_templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, List[str]]:
        """
        Initialize counter-stereotype prompt templates.

        Returns:
            Dictionary of bias type to counter-stereotype templates
        """
        templates = {
            'gender': [
                "Avoid gender stereotypes. Consider diverse gender representations.",
                "Ensure gender-neutral and inclusive language in your response.",
                "Challenge traditional gender role assumptions."
            ],
            'race': [
                "Avoid racial stereotypes and biases. Treat all races with equal respect.",
                "Provide culturally sensitive and inclusive responses.",
                "Recognize diversity within racial and ethnic groups."
            ],
            'occupation': [
                "Avoid occupational stereotypes. Any gender can pursue any career.",
                "Challenge traditional career-gender associations.",
                "Recognize diverse representation across all professions."
            ],
            'general': [
                "Provide fair, unbiased, and inclusive responses.",
                "Avoid perpetuating stereotypes or harmful generalizations.",
                "Consider diverse perspectives and experiences."
            ]
        }
        return templates

    def add_counterstereotype_instruction(self,
                                         prompt: str,
                                         bias_type: str = 'general') -> str:
        """
        Add counter-stereotype instruction to prompt.

        Args:
            prompt: Original prompt
            bias_type: Type of bias to address

        Returns:
            Modified prompt with counter-stereotype instruction
        """
        templates = self.counter_stereotype_templates.get(
            bias_type,
            self.counter_stereotype_templates['general']
        )

        instruction = "IMPORTANT: " + " ".join(templates) + "\n\n"
        return instruction + prompt

    def detect_stereotype_category(self, prompt: str) -> str:
        """
        Detect the stereotype category in a prompt.

        Args:
            prompt: Input prompt

        Returns:
            Detected category ('gender', 'race', 'occupation', or 'general')
        """
        prompt_lower = prompt.lower()

        gender_keywords = ['man', 'woman', 'male', 'female', 'he', 'she', 'boy', 'girl']
        race_keywords = ['black', 'white', 'asian', 'hispanic', 'african', 'race', 'ethnic']
        occupation_keywords = ['doctor', 'nurse', 'engineer', 'teacher', 'ceo', 'job', 'career', 'profession']

        if any(keyword in prompt_lower for keyword in gender_keywords):
            return 'gender'
        elif any(keyword in prompt_lower for keyword in race_keywords):
            return 'race'
        elif any(keyword in prompt_lower for keyword in occupation_keywords):
            return 'occupation'
        else:
            return 'general'

    def apply_counterstereotype_prompting(self,
                                         prompt: str,
                                         auto_detect: bool = True) -> str:
        """
        Apply counter-stereotype prompting technique.

        Args:
            prompt: Original prompt
            auto_detect: Automatically detect stereotype category

        Returns:
            Modified prompt with counter-stereotype instruction
        """
        if auto_detect:
            bias_type = self.detect_stereotype_category(prompt)
        else:
            bias_type = 'general'

        return self.add_counterstereotype_instruction(prompt, bias_type)

    def create_balanced_prompt(self, prompt: str, examples: List[str] = None) -> str:
        """
        Create a balanced prompt with diverse examples.

        Args:
            prompt: Original prompt
            examples: Optional list of balanced examples

        Returns:
            Enhanced prompt with balanced examples
        """
        base_instruction = (
            "Provide a balanced, fair, and inclusive response. "
            "Consider multiple perspectives and avoid stereotypes.\n\n"
        )

        enhanced_prompt = base_instruction + prompt

        if examples:
            enhanced_prompt += "\n\nConsider these balanced examples:\n"
            for i, example in enumerate(examples, 1):
                enhanced_prompt += f"{i}. {example}\n"

        return enhanced_prompt

    def evaluate_stereotype_presence(self, text: str) -> Dict[str, any]:
        """
        Evaluate presence of stereotypes in text.

        Args:
            text: Text to evaluate

        Returns:
            Dictionary with stereotype indicators
        """
        text_lower = text.lower()

        stereotype_indicators = {
            'gender_stereotype_keywords': [
                'all women', 'all men', 'typical woman', 'typical man',
                'women are', 'men are', 'women always', 'men always'
            ],
            'race_stereotype_keywords': [
                'all blacks', 'all whites', 'all asians',
                'typical black', 'typical white', 'typical asian'
            ],
            'occupation_stereotype_keywords': [
                'male doctor', 'female nurse', 'male engineer',
                'female teacher', 'male ceo', 'female secretary'
            ]
        }

        results = {
            'has_gender_stereotypes': False,
            'has_race_stereotypes': False,
            'has_occupation_stereotypes': False,
            'detected_phrases': []
        }

        for category, keywords in stereotype_indicators.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if 'gender' in category:
                        results['has_gender_stereotypes'] = True
                    elif 'race' in category:
                        results['has_race_stereotypes'] = True
                    elif 'occupation' in category:
                        results['has_occupation_stereotypes'] = True

                    results['detected_phrases'].append(keyword)

        results['has_stereotypes'] = any([
            results['has_gender_stereotypes'],
            results['has_race_stereotypes'],
            results['has_occupation_stereotypes']
        ])

        return results


class FairnessEnhancer:
    """Enhances response fairness through post-processing."""

    def __init__(self):
        """Initialize fairness enhancer."""
        self.mitigator = StereotypeMitigator()

    def enhance_response(self, response: str) -> Dict:
        """
        Enhance response for fairness.

        Args:
            response: Original response

        Returns:
            Dictionary with enhancement results
        """
        stereotype_eval = self.mitigator.evaluate_stereotype_presence(response)

        return {
            'original_response': response,
            'has_stereotypes': stereotype_eval['has_stereotypes'],
            'stereotype_details': stereotype_eval,
            'needs_revision': stereotype_eval['has_stereotypes']
        }


if __name__ == "__main__":
    print("Stereotype Mitigation Module")

    # Example usage
    mitigator = StereotypeMitigator()

    test_prompt = "What do women typically do for work?"
    print(f"\nOriginal prompt: {test_prompt}")

    detected_category = mitigator.detect_stereotype_category(test_prompt)
    print(f"Detected category: {detected_category}")

    modified_prompt = mitigator.apply_counterstereotype_prompting(test_prompt)
    print(f"\nModified prompt:\n{modified_prompt}")

    test_response = "Women typically work as nurses and teachers."
    evaluation = mitigator.evaluate_stereotype_presence(test_response)
    print(f"\nStereotype evaluation: {evaluation}")
