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
who_data_assessment = who_data[who_data["InformationType"] == "Assessment"].head(20)

# Extract cause-effect pairs
if not os.path.exists("./data/extracted_cause_effect.csv"):
    who_data_assessment["CauseEffectPairs"] = who_data_assessment.apply(
        lambda row: str(extractor.forward(row["Text"])), axis=1
    )
    who_data_assessment.to_csv("./data/extracted_cause_effect.csv", index=False)

# Display extracted results
st.subheader("Extracted Cause-Effect Pairs")
st.dataframe(who_data_assessment[who_data_assessment["CauseEffectPairs"].notnull()])
