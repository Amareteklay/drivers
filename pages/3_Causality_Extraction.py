# pages/3_Causality_Extraction.py

import streamlit as st
import pandas as pd
import dspy
from typing import List, Tuple
from config import configure_lm
from utils import chunk_text

# Configure the language model using our centralized config
lm = configure_lm()

st.title("Causality Extraction")

# Load the dataset (adjust the file path as needed)
who_data = pd.read_csv("./data/corpus.csv")
who_data_assessment = who_data[who_data["InformationType"] == "Assessment"]

st.dataframe(who_data_assessment.head(10))

# Define the DSPy signature for extracting cause-effect pairs
class CauseEffectExtractionSignature(dspy.Signature):
    """
    Extract cause and effect pairs from the provided text.
    """
    text = dspy.InputField(desc="Input text with potential cause-effect relationships.")
    cause = dspy.OutputField(desc="Identified cause (e.g., 'heavy rainfall').")
    effect = dspy.OutputField(desc="Identified effect (e.g., 'flooded streets').")

# Define the extraction module, keeping all extraction logic here.
class CauseEffectExtractionModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(CauseEffectExtractionSignature)
    
    def forward(self, text: str) -> List[Tuple[str, str]]:
        # Break the input text into smaller chunks using our helper from utils.py
        chunks = chunk_text(text)
        cause_effect_pairs = []
        for chunk in chunks:
            pair = self.extract_cause_effect(chunk)
            if pair:
                cause_effect_pairs.append(pair)
        return cause_effect_pairs
    
    def extract_cause_effect(self, text: str) -> Tuple[str, str]:
        # Construct a few-shot prompt with examples
        prompt = f"""
        Identify the cause and effect in the following sentence.

        Example 1:
        Text: "Due to heavy rainfall, the streets were flooded."
        Cause: heavy rainfall
        Effect: streets were flooded

        Example 2:
        Text: "Because of the power outage, all the lights went out."
        Cause: power outage
        Effect: lights went out

        Example 3:
        Text: "The company lost money since customers stopped buying their products."
        Cause: customers stopped buying their products
        Effect: company lost money

        Now, extract the cause and effect from the following sentence:
        Text: "{text}"
        Cause:
        """
        response = self.predict(text=prompt)
        if response and hasattr(response, 'cause') and hasattr(response, 'effect'):
            return response.cause, response.effect
        else:
            return None

# Initialize the extraction module
extractor = CauseEffectExtractionModule()

# Function to apply the extractor to each row of the dataset
def extract_from_row(row):
    text = row['Text']  # Adjust the column name if necessary
    return extractor.forward(text)

# Apply the extraction function to the first 10 rows and store the results in a new column
who_data_assessment['CauseEffectPairs'] = who_data_assessment.head(10).apply(extract_from_row, axis=1)

st.title("WHO Data Assessment: Causality Extraction")
for index, row in who_data_assessment.head(10).iterrows():
    st.write(f"Text: {row['Text']}")
    st.write("Extracted Cause-Effect Pairs:")
    for pair in row['CauseEffectPairs']:
        if pair:
            cause, effect = pair
            st.write(f"- Cause: {cause}")
            st.write(f"- Effect: {effect}")
    st.write("---")
