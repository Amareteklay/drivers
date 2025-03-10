import dspy
from utils import chunk_text_by_sentences

class CauseEffectExtractionSignature(dspy.Signature):
    """ A DSPy signature for extracting cause-effect pairs from text. """
    text = dspy.InputField(desc="Input text with potential cause-effect relationships.")
    cause = dspy.OutputField(desc="Identified cause.")
    effect = dspy.OutputField(desc="Identified effect.")

class CauseEffectExtractionModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(signature=CauseEffectExtractionSignature)

    def forward(self, text: str):
        chunks = chunk_text_by_sentences(text)
        cause_effect_pairs = []
        for chunk in chunks:
            pair = self.extract_cause_effect(chunk)
            if pair:
                cause, effect = pair
                cause_effect_pairs.append({"cause": cause, "effect": effect})
        return cause_effect_pairs

    def extract_cause_effect(self, text: str):
        response = self.predict(text=text)
        if response and hasattr(response, "cause") and hasattr(response, "effect"):
            return response.cause, response.effect
        return None
