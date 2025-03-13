import streamlit as st
import pandas as pd
import os
import ast
from config import configure_lm
from models.causal_extraction import CauseEffectExtractionModule
from models.validation import evaluate_model
from prompts.trainset import trainset

lm = configure_lm()
extractor = CauseEffectExtractionModule()
st.title("Causal Extraction")

# Evaluate model
accuracy, df_results = evaluate_model(trainset, extractor)
st.subheader("Model Evaluation Results")
st.write(f"Validation Accuracy: {accuracy:.2%}")
st.dataframe(df_results)

# Load dataset
who_data = pd.read_csv("./data/corpus.csv")
# TODO Test for the first 20 articles, to be run on the full dataset later
who_data_assessment = who_data[who_data["InformationType"] == "Assessment"].head(10)

# Path to extracted cause-effect pairs
extracted_file_path = "./data/output/extracted_cause_effect.csv"

# If extracted data exists, load it
if os.path.exists(extracted_file_path):
    st.write("Loading extracted cause-effect pairs from saved file...")
    who_data_assessment = pd.read_csv(extracted_file_path)  # Load directly
else:
    st.write("Extracting cause-effect pairs...")

    # Function to extract cause-effect pairs safely
    def extract_cause_effect(text):
        cause_effect_pairs = extractor.forward(
            text
        )  # Calls forward() which returns a list
        return (
            str(cause_effect_pairs)
            if cause_effect_pairs
            else "No cause-effect pairs found"
        )

    # Apply extraction
    who_data_assessment["CauseEffectPairs"] = (
        who_data_assessment["Text"].head(10).apply(extract_cause_effect)
    )

    # Save extracted results
    who_data_assessment.to_csv(extracted_file_path, index=False)
    st.write("Extraction complete. Data saved.")

# Display extracted results
st.subheader("Extracted Cause-Effect Pairs")
st.dataframe(who_data_assessment.head(10))
