import streamlit as st
import pandas as pd
import dspy

# Configure the language model
lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

st.title("Categorization")

# Load the Excel file
driver_cat = pd.read_excel(
    "./data/drivers.xlsx", 
    sheet_name="V2_Peter_PSJ_EP_PSJ_EP"
)

# Display the DataFrame in Streamlit
st.title("Driver Categories")


# Import result_df_31 Oct.csv
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
            "Cause_category_by_OpenAI",
        ]
    )
)

st.write("Below is the content of the `result_df` dataset:")
st.dataframe(result_df)  # Displays the DataFrame interactively


# Select only 10 rows for processing
result_df_sample = result_df.head(10).copy()

#st.dataframe(result_df_sample)

# Define the Categorization class
class Categorization(dspy.Signature):
    """Categorize text based on predefined categories and consolidate name mapping."""
    
    text = dspy.InputField()
    consolidate_name = dspy.OutputField(desc="A short, general name for the text topic")
    category = dspy.OutputField(desc="The broader category that best fits the text")

# Define the categorization predictor.
categorize_text = dspy.Predict(Categorization)

# Category examples for few-shot learning
category_examples = """
- Text: 'Socio-economic , socioeconomic change' -> consolidate_name: economy misc., category: Economy
- Text: 'shift from cold to hotter temperature' -> consolidate_name: temperature shift (cold to hotter), category: Climate and weather
- Text: 'Changes in demand for bushmeat' -> consolidate_name: bushmeat demand change, category: Wildlife
- Text: 'contact with infected poultry or environments that have been contaminated' -> consolidate_name: 'poultry exposure', category: Disease transmission
- Text: 'close contact with A(H5N1)-infected live or dead birds or mammals' -> consolidate_name: 'animal exposure', category: Disease transmission
"""

# Process each row in the sample dataframe
consolidate_names = []
categories = []

for cause_text in result_df_sample["Cause_by_OpenAI"]:
    prompt = (
        "Analyze the following text and map it to a predefined category from the list below. "
        "Return your answer as a valid JSON object with keys 'consolidate_name' and 'category'.\n\n"
        "Categories and examples:\n"
        f"{category_examples}\n\n"
        "If none of the above applies, please return {\"consolidate_name\": \"Undefined\", \"category\": \"Undefined\"}.\n\n"
        "Now, analyze the following text:\n"
        f"'{cause_text}'\n"
    )

    pred = categorize_text(text=prompt)

    if pred:
        consolidate_names.append(pred.consolidate_name)
        categories.append(pred.category)
    else:
        consolidate_names.append("Undefined")
        categories.append("Undefined")

# Add the new predictions to the DataFrame
result_df_sample["Consolidate_Name"] = consolidate_names
result_df_sample["Predicted_Category"] = categories

st.write('_'*50)
st.dataframe(result_df_sample)