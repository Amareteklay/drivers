# pages/4_Categorization.py

import streamlit as st
import pandas as pd
import dspy
import re
from typing import List
from config import configure_lm  # Import LM configuration from config.py

# Configure the language model using our config settings
lm = configure_lm()

st.title("Cause Categorization")

# --- Load and prepare the drivers data ---
driver_cat = pd.read_excel("./data/drivers.xlsx", sheet_name="V2_Peter_PSJ_EP_PSJ_EP")
driver_categories = driver_cat["Consolidated Name"].dropna().unique().tolist()
driver_categories = [str(cat).strip() for cat in driver_categories]

# Build regex pattern for matching categories
category_pattern = re.compile(
    r'\b(' + '|'.join(re.escape(cat) for cat in driver_categories) + r')\b', 
    flags=re.IGNORECASE
)

# --- Load and prepare the results data ---
result_df = pd.read_csv("./data/result_df_31_Oct.csv").rename(columns={
    "Cause": "Cause_by_OpenAI",
    "Effect": "Effect_by_OpenAI",
    "Cause_category": "Cause_category_by_OpenAI",
    "Effect_category": "Effect_category_by_OpenAI",
}).filter([
    "DonId",
    "Cause_by_OpenAI",
    "Effect_by_OpenAI",
    "Cause_category_by_OpenAI",
    "Effect_category_by_OpenAI",
    "Raw_Text",
])

# --- DSPy Signature for Categorization ---
class CauseCategorizationSignature(dspy.Signature):
    """Categorize cause text into one of the predefined driver categories."""
    cause_text = dspy.InputField(desc="Cause text from causal extraction")
    driver_category = dspy.OutputField(desc=f"Category from: {', '.join(driver_categories)}") # Use Literal here

# --- Category Validator Module ---
class CategoryValidator(dspy.Module):
    def __init__(self, categories: List[str]):
        super().__init__()
        self.categories = categories
        self.normalized_categories = {cat.lower(): cat for cat in categories}
        
    def forward(self, output: str) -> str:
        # Direct match
        if output in self.categories:
            return output
        # Normalized match
        normalized_output = output.strip().lower()
        for norm_cat, original_cat in self.normalized_categories.items():
            if norm_cat == normalized_output:
                return original_cat
        # Pattern matching via regex
        matches = category_pattern.findall(output)
        if matches:
            return matches[0].strip()
        return "Uncategorized"

# --- DSPy Module for Cause Categorization ---
class CauseCategorizationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(CauseCategorizationSignature)
        self.validator = CategoryValidator(driver_categories)

    def forward(self, cause_text: str) -> str:
        # Construct structured prompt
        prompt = (
            f"Analyze this cause text and categorize it EXACTLY as one of: {', '.join(driver_categories)}.\n"
            f"Text: {cause_text}\n"
            "Respond ONLY with the category name from the list. If unsure, say 'Uncategorized'."
        )
        
        response = self.predict(cause_text=prompt)
        # Extract the category output (assuming the response contains a colon-separated format)
        raw_category = response.driver_category.split(':')[-1].strip() if hasattr(response, 'driver_category') else ""
        # Validate and normalize the response
        validated_category = self.validator(raw_category)
        return validated_category

# --- Processing Pipeline ---
def process_dataframe(df: pd.DataFrame, categorizer: CauseCategorizationModule) -> pd.DataFrame:
    df = df.copy()
    progress_bar = st.progress(0)
    
    for idx, row in enumerate(df.itertuples()):
        df.at[row.Index, 'Cause_driver_category'] = categorizer.forward(row.Cause_by_OpenAI)
        progress_bar.progress((idx + 1) / len(df))
        
    return df

# --- Streamlit Interface ---
cause_categorizer = CauseCategorizationModule()

st.header("Data Preview")
st.subheader("Driver Categories")
st.write(f"Unique categories ({len(driver_categories)}):")
st.dataframe(driver_categories)

st.subheader("Input Data (Last 10 Rows)")
st.dataframe(result_df.head(20))

if st.button("Run Categorization"):
    st.header("Categorization Results")
    with st.spinner("Processing causes..."):
        result_df_categorized = process_dataframe(result_df.head(20), cause_categorizer)
    st.dataframe(result_df_categorized[['Cause_by_OpenAI', 'Cause_driver_category']])
    
    st.subheader("Category Distribution")
    category_counts = result_df_categorized['Cause_driver_category'].value_counts()
    st.bar_chart(category_counts)
