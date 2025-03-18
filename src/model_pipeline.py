import pandas as pd
import streamlit as st
import dspy
import random
from dspy.teleprompt import LabeledFewShot
from dspy.evaluate.evaluate import Evaluate
from config import configure_lm 
from src.data_utils import chunk_text_by_sentences

class CauseEffectSignature(dspy.Signature):
    text = dspy.InputField(desc="Paragraph containing cause/effect information")
    cause = dspy.OutputField(desc="Extracted cause(s)")
    effect = dspy.OutputField(desc="Extracted effect(s)")

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
    """
    lm = _initialize_lm()
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
    Runs inference on each row's full text.
    """
    results = []
    for idx, row in df.iterrows():
        text_val = row[text_col]
        prediction = model(text=text_val)
        results.append({
            'OriginalIndex': idx,
            'Text': text_val,
            'PredictedCause': prediction.cause,
            'PredictedEffect': prediction.effect
        })
    return pd.DataFrame(results)

def predict_cause_effect_with_chunking(df: pd.DataFrame, model, text_col='Text', chunk_size=4, overlap=1) -> pd.DataFrame:
    """
    Runs inference on each row's text after splitting it into chunks based on sentences.
    Each chunk is processed individually and the results are aggregated.
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
                'PredictedCause': prediction.cause,
                'PredictedEffect': prediction.effect
            })
    return pd.DataFrame(results)
