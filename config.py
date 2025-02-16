# config.py

import dspy
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="env/.env")

# Define LM configuration parameters
LM_MODEL = "ollama_chat/llama3.2"
OPEN_AI_MODEL = "openai/o3-mini-2025-01-31"
LM_API_BASE = "http://localhost:11434"
LM_API_KEY = os.getenv("OPENAI_API_KEY")  # Add your API key if needed


def configure_lm(model=["llama", "openai"]):
    """
    Configure and return the language model.
    """

    if "llama" in model:
        lm = dspy.LM(LM_MODEL, api_base=LM_API_BASE, api_key=LM_API_KEY)
    elif "openai" in model:
        lm = dspy.LM(OPEN_AI_MODEL, api_base=LM_API_BASE, api_key=LM_API_KEY)

    dspy.configure(lm=lm)
    return lm
