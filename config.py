
import dspy

# Define LM configuration parameters
LM_MODEL = 'ollama_chat/llama3.2'
LM_API_BASE = 'http://localhost:11434'
LM_API_KEY = ''  # Add your API key if needed

def configure_lm():
    """
    Configure and return the language model.
    """
    lm = dspy.LM(LM_MODEL, api_base=LM_API_BASE, api_key=LM_API_KEY)
    dspy.configure(lm=lm)
    return lm
