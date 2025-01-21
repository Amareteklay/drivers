import streamlit as st
from src.causality import CausalChain
from src.prompt import PromptDesigner

st.title("Causality Extraction")

text = st.text_area("Enter text for analysis")
if st.button("Extract"):
    prompt_designer = PromptDesigner()
    causal_chain = CausalChain(dataframe=None, prompt_designer=prompt_designer)
    results = causal_chain.extract_cause_effect_openai(text, prompt_parts={}, don_id=None)
    st.write(results)
