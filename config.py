import dspy
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv("env/.env")

# Define LM configuration parameters
DEFAULT_LM_MODEL = "openai/o3-mini"
ALTERNATIVE_LM_MODEL = "ollama_chat/llama3.2"
ALTERNATIVE_LM_API_BASE = "http://localhost:11434"
DEFAULT_LM_API_KEY = os.getenv("OPENAI_API_KEY")


def configure_lm(model: str = "openai"):
    """
    Configure and return the language model based on the given model_type.
    model can be either "openai" or "llama".
    """
    if model.lower() == "openai":
        # max_tokens=5000, temperature=1.0: only specify when using o3-mini, bug of dpsy
        lm = dspy.LM(
            DEFAULT_LM_MODEL,
            api_key=DEFAULT_LM_API_KEY,
            max_tokens=5000,
            temperature=1.0,
        )
    else:
        lm = dspy.LM(ALTERNATIVE_LM_MODEL, api_base=ALTERNATIVE_LM_API_BASE, api_key="")

    dspy.configure(lm=lm)
    return lm
