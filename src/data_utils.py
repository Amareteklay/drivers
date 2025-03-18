# src/data_utils.py
import re
import pandas as pd

def load_train_data(json_path: str) -> pd.DataFrame:
    """
    Read the JSON file containing training data.
    Expected format: a list of dictionaries with keys like 'Text', 'Cause', and 'Effect'.
    Returns a DataFrame for further processing.
    """
    df = pd.read_json(json_path)
    # Optional: add cleaning or filtering steps (e.g., handling missing values)
    return df

def load_inference_data(csv_path: str) -> pd.DataFrame:
    """
    Read the CSV file containing inference data.
    Expected format: a CSV with at least a 'Text' column.
    Returns a DataFrame for further processing.
    """
    df = pd.read_csv(csv_path)
    # Optional: add cleaning or filtering steps (e.g., drop duplicates, handle missing values)
    return df

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> list:
    """
    Splits the text into chunks of a specified size with a given overlap.
    
    Args:
        text (str): The text to be split.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of overlapping words between consecutive chunks.
        
    Returns:
        List[str]: A list of text chunks.
    """
    if not isinstance(text, str):
        text = str(text)  # Convert to string if it's not
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def chunk_text_by_sentences(text: str, chunk_size: int = 4, overlap: int = 1) -> list:
    """
    Splits the text into chunks of a specified number of sentences with a given overlap.
    
    Args:
        text (str): The text to be split.
        chunk_size (int): Number of sentences per chunk.
        overlap (int): Number of overlapping sentences between consecutive chunks.
        
    Returns:
        List[str]: A list of text chunks.
    """
    # Split text into sentences using regex (basic sentence splitting)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i:i + chunk_size]
        chunks.append(' '.join(chunk))
        
        # Move index forward but maintain overlap
        i += chunk_size - overlap
    
    return chunks