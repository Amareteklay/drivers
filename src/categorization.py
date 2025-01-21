import pandas as pd

def format_prompt(text: str, category_dict: dict) -> str:
    prompt = f"...\n{text}\n..."
    return prompt

def get_summary_with_prelist(text: str, category_dict: dict):
    # OpenAI categorization logic
    pass

def categorize_text(row: pd.Series, column_name: str, category_dict: dict) -> pd.Series:
    consolidate_name, category = get_summary_with_prelist(row[column_name], category_dict)
    return pd.Series({
        f"{column_name}_consolidated": consolidate_name,
        f"{column_name}_group": category,
    })
