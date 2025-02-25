import streamlit as st
import pandas as pd
import dspy
import os
import re
from typing import List, Literal
from config import configure_lm  # Import LM configuration from config.py
from sentence_transformers import SentenceTransformer, util

# Configure the language model using our config settings
lm = configure_lm()

st.title("Cause Categorization")

# Path for saving extracted results
saved_results_path = "./data/mapped_categories.csv"

# --- Load and prepare the drivers data ---
driver_cat = pd.read_excel("./data/drivers.xlsx", sheet_name="V2_Peter_PSJ_EP_PSJ_EP")
driver_categories = driver_cat["Consolidated Name"].dropna().unique().tolist()
st.write("Original driver categories:", driver_categories)
driver_categories = [str(cat).strip() for cat in driver_categories]
st.write("Cleaned driver categories:", driver_categories)
# Build regex pattern for matching categories
category_pattern = re.compile(
    r"\b(" + "|".join(re.escape(cat) for cat in driver_categories) + r")\b",
    flags=re.IGNORECASE,
)

# --- Load and prepare the results data ---
result_df = (
    pd.read_csv("./data/result_df_31_Oct.csv")
    .rename(
        columns={
            "Cause": "Cause_by_OpenAI",
            "Effect": "Effect_by_OpenAI",
            "Cause_category": "Cause_category_by_OpenAI",
            "Effect_category": "Effect_category_by_OpenAI",
        }
    )
    .filter(
        [
            "DonId",
            "Cause_by_OpenAI",
            "Effect_by_OpenAI",
            "Cause_category_by_OpenAI",
            "Effect_category_by_OpenAI",
            "Raw_Text",
        ]
    )
)

# --- DSPy Signature for Categorization ---
class CauseCategorizationSignature(dspy.Signature):
    """
    Categorize cause text into one of the predefined driver categories.
    """
    cause_text = dspy.InputField(desc="Cause text from causal extraction")
    driver_category = dspy.OutputField(
        desc=f"Category from: {', '.join(driver_categories)}",
        type=Literal[tuple(driver_categories)]  # Enforcing a Literal output
    )

# --- Category Validator Module with Semantic Similarity ---
class CategoryValidator(dspy.Module):
    def __init__(self, categories: List[str]):
        """
        Initialize the category validator module.
        Parameters:
        categories (List[str]): List of valid categories to validate against.
        """
        super().__init__()
        self.categories = categories
        self.normalized_categories = {cat.lower(): cat for cat in categories}
        # Initialize the semantic similarity model
        st.write("Loading semantic similarity model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        st.write("Model loaded.")
        # Precompute embeddings for driver categories
        st.write("Computing category embeddings...")
        self.category_embeddings = self.model.encode(categories, convert_to_tensor=True)
        st.write("Category embeddings computed.")

    def forward(self, output: str) -> str:
        # Direct match
        if output in self.categories:
            st.write("Direct match found:", output)
            return output
        # Normalized match
        normalized_output = output.strip().lower()
        for norm_cat, original_cat in self.normalized_categories.items():
            if norm_cat == normalized_output:
                st.write("Normalized match found:", original_cat)
                return original_cat
        # Regex matching fallback
        matches = category_pattern.findall(output)
        if matches:
            st.write("Regex match found:", matches[0].strip())
            return matches[0].strip()
        # Semantic similarity matching
        st.write("Performing semantic similarity matching for output:", output)
        output_embedding = self.model.encode(output, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(output_embedding, self.category_embeddings)
        best_score, best_idx = cosine_scores[0].max(dim=0)
        st.write("Semantic similarity score:", best_score.item(), "for category:", self.categories[best_idx])
        # Set a threshold for acceptance (tune as needed)
        if best_score > 0.5:
            st.write("Semantic match accepted:", self.categories[best_idx])
            return self.categories[best_idx]
        st.write("No suitable match found. Returning 'Uncategorized'")
        return "Uncategorized"

# --- DSPy Module for Cause Categorization ---
class CauseCategorizationModule(dspy.Module):
    def __init__(self):
        """
        Initialize the cause categorization module.
        This module is a wrapper around ChainOfThought that validates the output category
        against the predefined driver categories.
        """
        super().__init__()
        self.predict = dspy.ChainOfThought(CauseCategorizationSignature)
        self.validator = CategoryValidator(driver_categories)

    def forward(self, cause_text: str) -> str:
        # Construct a strict prompt to guide the LLM output
        prompt = (
            f"Analyze this cause text and categorize it EXACTLY as one of: {', '.join(driver_categories)}.\n"
            f"Text: {cause_text}\n"
            "Respond ONLY with the category name from the list. Do not include any extra text. "
            "If unsure, respond with 'Uncategorized'."
        )
        st.write("Prompt sent to LLM:", prompt)
        response = self.predict(cause_text=prompt)
        raw_response = response.driver_category if hasattr(response, "driver_category") else ""
        st.write("Raw LLM response:", raw_response)
        # Extract the category output (assuming colon-separated format)
        raw_category = (
            raw_response.split(":")[-1].strip() if raw_response else ""
        )
        st.write("Extracted raw category:", raw_category)
        # Validate and normalize the response
        validated_category = self.validator(raw_category)
        st.write("Validated category:", validated_category)
        return validated_category

# --- Processing Pipeline ---
def process_dataframe(df: pd.DataFrame, categorizer: CauseCategorizationModule) -> pd.DataFrame:
    """
    Process a given dataframe by categorizing the causes using the provided categorizer module.
    Parameters:
    df (pd.DataFrame): DataFrame to be processed.
    categorizer (CauseCategorizationModule): Module to be used for categorization.
    Returns:
    pd.DataFrame: Processed DataFrame with the categorized causes in the 'Cause_driver_category' column.
    """
    df = df.copy()
    progress_bar = st.progress(0)
    for idx, row in enumerate(df.itertuples()):
        st.write(f"Processing row {idx}: {row.Cause_by_OpenAI}")
        df.at[row.Index, "Cause_driver_category"] = categorizer.forward(row.Cause_by_OpenAI)
        progress_bar.progress((idx + 1) / len(df))
    return df

# --- Streamlit Interface ---
cause_categorizer = CauseCategorizationModule()

st.header("Data Preview")
st.subheader("Driver Categories")
st.write(f"Unique categories ({len(driver_categories)}):")
st.dataframe(driver_categories)

st.subheader("Input Data (First 10 Rows)")
st.dataframe(result_df.head(10))

# --- Load saved categorization results if available ---
if os.path.exists(saved_results_path):
    st.write("Loading previously saved categorization results...")
    result_df_categorized = pd.read_csv(saved_results_path)
    st.dataframe(result_df_categorized[["Cause_by_OpenAI", "Cause_driver_category"]])
else:
    result_df_categorized = None

# --- Run Categorization Only If Needed ---
if st.button("Run Categorization"):
    st.header("Categorization Results")
    if result_df_categorized is not None:
        st.write("Using previously saved categorization results.")
    else:
        st.write("No saved results found. Running categorization...")
        with st.spinner("Processing causes..."):
            result_df_categorized = process_dataframe(result_df.head(10), cause_categorizer)
        # Save the categorized results for future use
        result_df_categorized.to_csv(saved_results_path, index=False)
        st.write("Categorization completed and results saved.")
    st.dataframe(result_df_categorized[["Cause_by_OpenAI", "Cause_driver_category"]])
    st.subheader("Category Distribution")
    category_counts = result_df_categorized["Cause_driver_category"].value_counts()
    st.bar_chart(category_counts)
