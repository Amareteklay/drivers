import pandas as pd
import streamlit as st
import dspy
import random
from dspy.teleprompt import LabeledFewShot
from dspy.evaluate.evaluate import Evaluate
from config import configure_lm 
from src.data_utils import chunk_text_by_sentences

# Updated Signature to match richer training data
class CauseEffectSignature(dspy.Signature):
    text = dspy.InputField(desc="Raw outbreak report text")
    marked_text = dspy.OutputField(desc="Text with inline annotations for cause/effect")
    extracted_pairs = dspy.OutputField(desc="Structured extracted cause-effect pairs (list of dicts)")

def _initialize_lm():
    """
    Helper function to configure the LM and update DSPy's settings.
    Returns the LM instance.
    """
    lm = configure_lm()
    dspy.settings.configure(lm=lm)
    return lm

def build_and_train_model(train_examples):
    """
    Builds and compiles the DSPy Chain-of-Thought model using a few-shot approach.
    Expects training examples to contain keys corresponding to the new signature (e.g., text, marked_text, extracted_pairs).
    """
    lm = _initialize_lm()
    st.write("Before training, dspy.settings.lm:", dspy.settings.lm)
    st.write("Before training, model.lm:", lm)  # This should be the same as dspy.settings.lm
    cot_model = dspy.ChainOfThought(CauseEffectSignature)
    cot_model.lm = lm  # explicitly assign the LM
    few_shot_optimizer = LabeledFewShot(k=min(7, len(train_examples)))
    few_shot_cause_effect = few_shot_optimizer.compile(
        student=cot_model,
        trainset=train_examples
    )
    return few_shot_cause_effect

def load_model(model_path: str):
    """
    Loads a saved DSPy model from disk and reassigns the LM.
    """
    lm = _initialize_lm()
    model = dspy.ChainOfThought(CauseEffectSignature)
    model.load(model_path)
    model.lm = lm
    st.write("After loading, dspy.settings.lm:", dspy.settings.lm)
    st.write("After loading, model.lm:", model.lm)
    return model

def predict_cause_effect(df: pd.DataFrame, model, text_col='Text') -> pd.DataFrame:
    """
    Runs inference on each row's full text and returns a DataFrame with:
    - The original text
    - The model's predicted marked text
    - The model's extracted cause/effect pairs.
    """
    results = []
    for idx, row in df.iterrows():
        text_val = row[text_col]
        prediction = model(text=text_val)
        results.append({
            'OriginalIndex': idx,
            'Text': text_val,
            'PredictedMarkedText': prediction.marked_text,
            'PredictedExtractedPairs': prediction.extracted_pairs
        })
    return pd.DataFrame(results)

def predict_cause_effect_with_chunking(df: pd.DataFrame, model, text_col='Text', chunk_size=4, overlap=1) -> pd.DataFrame:
    """
    Splits each text into chunks (using sentence-based chunking), runs predictions on each chunk,
    and aggregates the results.
    """
    results = []
    for idx, row in df.iterrows():
        text_val = row[text_col]
        # Split text into chunks (e.g., 4 sentences per chunk with an overlap of 1 sentence)
        chunks = chunk_text_by_sentences(text_val, chunk_size=chunk_size, overlap=overlap)
        for chunk in chunks:
            prediction = model(text=chunk)
            results.append({
                'OriginalIndex': idx,
                'Text': text_val,
                'Chunk': chunk,
                'PredictedMarkedText': prediction.marked_text,
                'PredictedExtractedPairs': prediction.extracted_pairs
            })
    return pd.DataFrame(results)
