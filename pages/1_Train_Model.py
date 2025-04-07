import streamlit as st
import pandas as pd
import random
from src.data_utils import load_train_data
from src.model_pipeline import build_and_train_model
import dspy

st.title("Train Cause/Effect Extraction Model")

st.write("1. Load training data from JSON.")
df_train = load_train_data("data/train_set.json")

st.write(df_train)

st.write("2. Convert DataFrame rows into DSPy.Example objects.")
train_examples = []
for i, row in df_train.iterrows():
    ex = dspy.Example(
        text=row["text"],
         marked_text=row["marked_text"],
        extracted_pairs=row["extracted_pairs"]
    ).with_inputs("text")
    train_examples.append(ex)

st.write(f"Loaded {len(train_examples)} training examples.")
st.write(train_examples)
st.write("3. Train model on button click.")

if st.button("Train Model"):
    model = build_and_train_model(train_examples)
    model.save("./models/cause_effect_model.json")
    st.success("Model trained and saved!")
