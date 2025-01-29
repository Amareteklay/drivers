import streamlit as st
import dspy


st.title("Causality Extraction")

# Configure the language model
lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class CauseEffectClassifier(dspy.Signature):
    """Identify cause and effect relationships in text."""
    
    text = dspy.InputField()
    cause = dspy.OutputField(desc="The cause phrase in the text (e.g., 'heavy rainfall')")
    effect = dspy.OutputField(desc="The effect phrase in the text (e.g., 'flooded streets')")

# Manually define a few-shot prompt
few_shot_examples = """
Identify the cause and effect in the following sentences.

Example 1:
Text: "Due to heavy rainfall, the streets were flooded."
Cause: heavy rainfall
Effect: flooded streets

Example 2:
Text: "Because of the power outage, all the lights went out."
Cause: power outage
Effect: lights went out

Example 3:
Text: "The company lost money since customers stopped buying their products."
Cause: customers stopped buying their products
Effect: company lost money

Now, extract the cause and effect from the following sentence:
"""

# Define the predictor.
extract_cause_effect = dspy.Predict(CauseEffectClassifier)

# User-provided input
test_text = "A lack of sleep led to poor concentration during the meeting."

# Call the predictor on the input with the few-shot prompt.
pred = extract_cause_effect(text=few_shot_examples + f"\nText: \"{test_text}\"\nCause:")

if pred is None:
    st.write("No cause-effect detected.")
else:
    # Print the identified cause and effect.
    st.write(f"Text: {test_text}")
    st.write(f"Predicted Cause: {pred.cause}")
    st.write(f"Predicted Effect: {pred.effect}")