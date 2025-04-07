# pages/3_Review_Results.py
import streamlit as st
import pandas as pd

st.title("Review/Feedback on Predictions")

# Define the path to the saved predictions CSV.
pred_csv_path = "./data/output/extracted_cause_effect.csv"

# Try to load the predictions from disk.
try:
    df_pred = pd.read_csv(pred_csv_path)
    st.write("### Preview of Predictions")
    st.dataframe(df_pred.head(20))
    
    # Show basic summary statistics.
    st.write("### Summary Metrics")
    st.write(f"Total records: {len(df_pred)}")
    
    if 'PredictedCause' in df_pred.columns:
        st.write("#### Count by Predicted Cause")
        st.dataframe(df_pred['PredictedCause'].value_counts().reset_index().rename(columns={'index': 'Cause', 'PredictedCause': 'Count'}))
    
    if 'PredictedEffect' in df_pred.columns:
        st.write("#### Count by Predicted Effect")
        st.dataframe(df_pred['PredictedEffect'].value_counts().reset_index().rename(columns={'index': 'Effect', 'PredictedEffect': 'Count'}))
    
    # Optionally, allow further interactive review:
    st.write("### Detailed Data")
    st.data_editor(df_pred, num_rows="dynamic")
    
except Exception as e:
    st.error(f"Error loading predictions from {pred_csv_path}: {e}")
