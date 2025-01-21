import dspy
from typing import List, Tuple

class CausalChain:
    def __init__(self, model_name: str = "ollama_chat/llama3.2", api_base: str = "http://localhost:11434"):
        """
        Initialize the causal chain class with the Llama language model.

        Args:
            model_name (str): Name of the Llama model to use.
            api_base (str): Base URL for the Llama API.
        """
        self.lm = dspy.LM(model_name, api_base=api_base)

    def extract_causality(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract cause-effect pairs from the given text using Llama.

        Args:
            text (str): Input text to analyze.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing cause and effect pairs.
        """
        # Define a DSPy prompt for causality extraction

        # Run the Llama model with the defined prompt
        response = self.lm(causality_prompt)
        output = response.strip().split("\n")

        # Parse the output into cause-effect pairs
        cause_effect_pairs = []
        for line in output:
            if "Cause:" in line and "Effect:" in line:
                cause = line.split("Cause:")[1].split("Effect:")[0].strip()
                effect = line.split("Effect:")[1].strip()
                cause_effect_pairs.append((cause, effect))

        return cause_effect_pairs
