import streamlit as st
import pandas as pd
import dspy
from config import configure_lm
from src.data_utils import load_inference_data  # now includes chunking functions if needed
from src.model_pipeline import load_model, predict_cause_effect, predict_cause_effect_with_chunking

st.title("Run Inference on New Data")

# --- Global LM configuration ---
# Configure the LM before anything else so that dspy.settings.lm is set.
lm = configure_lm()
dspy.settings.configure(lm=lm)
st.write("Global LM configured:", lm)

# Initialize model in session state if not already present
if "model" not in st.session_state:
    st.session_state.model = None

# 1. Load the trained model and store it in session state.
if st.button("Load Trained Model"):
    st.session_state.model = load_model("./models/cause_effect_model.json")
    st.success("Model loaded successfully!")

# 2. Load inference data from a CSV file.
df_infer = load_inference_data("./data/corpus.csv")
if "Text" not in df_infer.columns:
    st.error("No 'Text' column in the CSV. Please check file format.")
else:
    # 3. Let the user select the prediction method.
    method = st.radio("Select Prediction Method", ("Full Text", "Chunked Text"))
    
    # 4. Run predictions based on the selected method.
    if st.button("Run Prediction"):
        if st.session_state.model is None:
            st.error("Please load the trained model first!")
        else:
            if method == "Full Text":
                results_df = predict_cause_effect(df_infer.head(5), st.session_state.model, text_col='Text')
            else:  # Chunked Text
                # Here, you can adjust the chunk size and overlap as needed.
                results_df = predict_cause_effect_with_chunking(df_infer.head(5), st.session_state.model, text_col='Text', chunk_size=4, overlap=1)
            
            st.write("Prediction Results Preview:")
            st.write(results_df.head())

            # Automatically save the results to disk for later review.
            results_path = "./data/output/extracted_cause_effect.csv"
            results_df.to_csv(results_path, index=False)
            st.success(f"Predictions saved to {results_path}")

            # Provide an option for the user to download the CSV.
            results_csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=results_csv,
                file_name="cause_effect_predictions.csv",
                mime="text/csv"
            )
