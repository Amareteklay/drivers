import streamlit as st
from src.categorization import categorize_text

st.title("Categorization")

text = st.text_area("Enter text for categorization")
categories = {"Transport": ["traffic"], "Environment": ["pollution"]}
if st.button("Categorize"):
    result = categorize_text({"text": text}, "text", categories)
    st.write(result)
