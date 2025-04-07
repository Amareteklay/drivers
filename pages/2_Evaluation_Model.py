# pages/3_Evaluate_Model.py
import streamlit as st
import pandas as pd
import dspy
from config import configure_lm
from src.data_utils import load_train_data
from src.model_pipeline import load_model
from src.evaluation import (
    extracted_pairs_exact_match,
    extracted_pairs_semantic_match,
    run_evaluation
)

st.title("Evaluate Model Performance")

# 1. Load a dev set for evaluation.
# Here we assume that your training JSON can also serve as a dev set.
df_dev = load_train_data("./data/train_set.json")
dev_examples = []
for idx, row in df_dev.iterrows():
    ex = dspy.Example(
        text=row["text"],
        marked_text=row["marked_text"],
        extracted_pairs=row["extracted_pairs"]
    ).with_inputs("text")
    dev_examples.append(ex)

st.write("Loaded", len(dev_examples), "development examples for evaluation.")

# 2. Load the saved model.
if st.button("Load Model for Evaluation"):
    model = load_model("./models/cause_effect_model.json")
    st.success("Model loaded successfully!")
    
    # 3. Create an evaluator using DSPy's Evaluate.
    evaluator = dspy.evaluate.evaluate.Evaluate(
        devset=dev_examples,
        display_progress=True,
        display_table=5
    )
    
    # 4. Run evaluation with the exact match metric.
    avg_exact, scores_exact = evaluator(
        model, metric=extracted_pairs_exact_match, return_all_scores=True
    )
    st.write("### Exact Match Evaluation")
    st.write("Average Exact Match Score:", avg_exact)
    st.write("Scores per example:", scores_exact)
    
    # 5. Run evaluation with the semantic similarity metric.
    avg_semantic, scores_semantic = evaluator(
        model, metric=extracted_pairs_semantic_match, return_all_scores=True
    )
    st.write("### Semantic Evaluation")
    st.write("Average Semantic Score:", avg_semantic)
    st.write("Scores per example:", scores_semantic)
