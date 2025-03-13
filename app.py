import streamlit as st

# App Title
st.set_page_config(page_title="Causal Extraction Dashboard", layout="wide")

# Overview Section
st.title("Drivers of EPPs Project")
st.markdown(
    """
    **Welcome to the Causal Extraction Dashboard!**  
    
    This project tries to extract cause-and-effect relationships from text and categorize the causes into different categories.

    The system is powered by **Declarative Self-Improving Python (DSPy)** to optimize prompt engineering.
    
    - **Goal**: Automatically detect causal relationships in textual data.

    - **Technology**: 
        - Uses DSPy for iterative improvement.
        - Use OpenAI o3-mini under the hood as the default LLM model 
        - It can be adapted easily to other LLMs, such as llama3.2 

    - **Features**:
        - **Casual Extraction:** Extracts causes and effects from input text.
        - **Drivers Mapping:** Classifies causes into predefined categories.

    Navigate to the different pages using the sidebar or the buttons below.
    """
)

# Navigation Buttons
st.write("## Navigate to Pages")
col1, col2 = st.columns(2)

with col1:
    if st.button("Causal Extraction"):
        st.switch_page("pages/1_Causality_Extraction.py")

with col2:
    if st.button("Drivers Mapping"):
        st.switch_page("pages/2_Mapping.py")

# Footer
st.markdown("---")
st.markdown("👨‍💻 **Developed by INFLUX Team**")
