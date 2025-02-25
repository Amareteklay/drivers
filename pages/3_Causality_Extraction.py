import streamlit as st
import pandas as pd
import dspy
import os
import ast
from typing import List, Tuple
from config import configure_lm
from utils import chunk_text
from prompts.few_shot_examples import few_shot_examples

# Configure the language model using our centralized config
lm = configure_lm()

st.title("Causality Extraction")

# Load the dataset (adjust the file path as needed)
who_data = pd.read_csv("./data/corpus.csv")
who_data_assessment = who_data[who_data["InformationType"] == "Assessment"]

st.dataframe(who_data_assessment.head(10))

# Path for saving extracted results
saved_results_path = "./data/extracted_cause_effect.csv"

class CauseEffectExtractionSignature(dspy.Signature):
    """
    A DSPy signature for extracting cause-effect pairs from text.
    
    Attributes:
        text (dspy.InputField): The input text that may contain cause-effect relationships.
        cause (dspy.OutputField): The extracted cause from the input text.
        effect (dspy.OutputField): The extracted effect from the input text.
    """
    text = dspy.InputField(desc="Input text with potential cause-effect relationships.")
    cause = dspy.OutputField(desc="Identified cause.")
    effect = dspy.OutputField(desc="Identified effect.")


# Define the extraction module using few-shot examples
class CauseEffectExtractionModule(dspy.Module):
    """
    A module for extracting cause-effect pairs from text using DSPy's ChainOfThought.
    
    Methods:
        forward(text: str) -> List[Tuple[str, str]]:
            Processes the input text, extracts, and returns a list of cause-effect pairs.
        extract_cause_effect(text: str) -> Tuple[str, str]:
            Extracts a cause-effect pair from a single text chunk using few-shot examples.
    """
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(
            signature=CauseEffectExtractionSignature,
            examples=few_shot_examples  # Using few-shot examples as dictionaries
        )

    def forward(self, text: str) -> List[Tuple[str, str]]:
        """
        Processes the input text to extract cause-effect pairs.
        
        The text is split into chunks (using `chunk_text`) and each chunk is processed to extract a cause-effect pair.
        
        Args:
            text (str): The input text containing potential cause-effect relationships.
        
        Returns:
            List[Tuple[str, str]]: A list of tuples, each containing a cause and its corresponding effect.
        """
        chunks = chunk_text(text)
        cause_effect_pairs = []
        for chunk in chunks:
            pair = self.extract_cause_effect(chunk)
            if pair:
                cause_effect_pairs.append(pair)
        return cause_effect_pairs

    def extract_cause_effect(self, text: str) -> Tuple[str, str]:
        """
        Extracts a cause-effect pair from the given text chunk using few-shot examples.
        
        Args:
            text (str): A text chunk with potential cause-effect information.
        
        Returns:
            Tuple[str, str]: A tuple containing the extracted cause and effect, or None if extraction fails.
        """
        # Create a minimal prompt; few-shot examples are integrated automatically
        prompt = f"Text: \"{text}\"\nCause:"
        response = self.predict(text=prompt)
        if response and hasattr(response, "cause") and hasattr(response, "effect"):
            return (response.cause, response.effect)
        else:
            return None

# Function to apply the extractor to each row of the dataset
def extract_from_row(row):
    text = row["Text"]  # Adjust the column name if necessary
    return extractor.forward(text)

# Check if saved results exist
if os.path.exists(saved_results_path):
    st.write("Loading saved extraction results...")
    who_data_assessment = pd.read_csv(saved_results_path)
else:
    extractor = CauseEffectExtractionModule()
    st.write("Running extraction...")
    who_data_assessment["CauseEffectPairs"] = who_data_assessment.head(10).apply(extract_from_row, axis=1)
    who_data_assessment.to_csv(saved_results_path, index=False)
    st.write("Extraction completed and results saved.")

# Display the extracted data
st.title("WHO Data Assessment: Causality Extraction")
st.write(who_data_assessment['CauseEffectPairs'])

for index, row in who_data_assessment.head(10).iterrows():
    st.write(f"Text: {row['Text']}")
    st.write("Extracted Cause-Effect Pairs:")
    pairs = row["CauseEffectPairs"]
    if isinstance(pairs, str):
        try:
            pairs = ast.literal_eval(pairs)
        except Exception as e:
            st.write("Error parsing CauseEffectPairs:", e)
            continue
    # If not a string, assume it's already a list.
    for pair in pairs:
        if pair:
            cause, effect = pair
            st.write(f"- Cause: {cause}")
            st.write(f"- Effect: {effect}")
    st.write("---")
