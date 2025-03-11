import dspy
from utils import chunk_text_by_sentences
import streamlit as st

class CauseEffectExtractionSignature(dspy.Signature):
    """ A DSPy signature for extracting cause-effect pairs from text. """
    text = dspy.InputField(desc="Input text with potential cause-effect relationships.")
    cause = dspy.OutputField(desc="A list of any identified causes.")
    effect = dspy.OutputField(desc="A list of any identified effects.")

class CauseEffectExtractionModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(signature=CauseEffectExtractionSignature)

    def forward(self, text: str):
        chunks = chunk_text_by_sentences(text)
        cause_effect_pairs = []

        for chunk in chunks:
            response = self.predict(text=chunk)  # Directly call predict()
            st.write(f"Chunk: {chunk}")

            if response and hasattr(response, "cause") and hasattr(response, "effect"):
                cause_effect_pairs.append({"cause": response.cause, "effect": response.effect})
                st.write(f"Cause: {response.cause}")
                st.write(f"Effect: {response.effect}")

        return cause_effect_pairs
