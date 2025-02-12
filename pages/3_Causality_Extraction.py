# pages/3_Causality_Extraction.py

import streamlit as st
import pandas as pd
import dspy
from typing import List, Tuple
from config import configure_lm
from utils import chunk_text

# Configure the language model using our centralized config
lm = configure_lm()

st.title("Causality Extraction")

# Load the dataset (adjust the file path as needed)
who_data = pd.read_csv("./data/corpus.csv")
who_data_assessment = who_data[who_data["InformationType"] == "Assessment"]

st.dataframe(who_data_assessment.head(20))

# Define the DSPy signature for extracting cause-effect pairs
class CauseEffectExtractionSignature(dspy.Signature):
    """
    Extract cause and effect pairs from the provided text.
    """
    text = dspy.InputField(desc="Input text with potential cause-effect relationships.")
    cause = dspy.OutputField(desc="Identified cause (e.g., 'heavy rainfall').")
    effect = dspy.OutputField(desc="Identified effect (e.g., 'flooded streets').")

# Define the extraction module, keeping all extraction logic here.
class CauseEffectExtractionModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(CauseEffectExtractionSignature)
    
    def forward(self, text: str) -> List[Tuple[str, str]]:
        # Break the input text into smaller chunks using our helper from utils.py
        chunks = chunk_text(text)
        cause_effect_pairs = []
        for chunk in chunks:
            pair = self.extract_cause_effect(chunk)
            if pair:
                cause_effect_pairs.append(pair)
        return cause_effect_pairs
    
    def extract_cause_effect(self, text: str) -> Tuple[str, str]:
        # Construct a few-shot prompt with examples
        prompt = f"""
        Below are some examples how causality can be reported in different forms:
        - Single cause, single effect (Type 1)

        Example 1: (C1) High population density and mobility in urban areas (C1) have facilitated (E1) the rapid spread of the virus (E1)". 

        Example 2: There is (C1) no vaccine for Influenza A(H1N1)v infection currently licensed for use in humans (C1). Seasonal influenza vaccines against human influenza viruses are generally not expected to protect people from (E1) infection with influenza viruses (E1) that normally circulate in pigs, but they can reduce severity.


        - Single cause, multiple effects (Type 2)

        Example 3: Several countries including Cameroon, Ethiopia, Haiti, Lebanon, Nigeria (north-east of the country), Pakistan, Somalia, Syria and the Democratic Republic of Congo (eastern part of the country) are in the midst of complex (C1) humanitarian crises (C1) with (E1) fragile health systems (E1), (E1) inadequate access to clean water and sanitation (E1) and have (E1) insufficient capacity to respond to the outbreaks (E1)

        - Multiple causes, single effect (Type 3)
        Example 4: Moreover, (C1) a low index of suspicion (C1), (C1) socio-cultural norms (C1), (C1) community resistance (C1), (C1) limited community knowledge regarding anthrax transmission (C1), (C1) high levels of poverty (C1) and (C1) food insecurity (C1), (C1) a shortage of available vaccines and laboratory reagents (C1), (C1) inadequate carcass disposal (C1) and (C1) decontamination practices (C1) significantly contribute to hampering (E1) the containment of the anthrax outbreak (E1).

        Example 5:
        The (E1) risk at the national level (E1) is assessed as 'High' due to the following:
        + In other parts of Timor-Leste (C1) health workers have limited knowledge dog bite and scratch case management (C1) including PEP and RIG administration
        + (C2) Insufficient stock of human rabies vaccines (C2) in the government health facilities.

        - Multiple causes, multiple effects (Type 4) - Chain of causalities
        The text may describe a chain of causality, where one effect becomes then the cause of another effect. To describe the chain, you should number the causes and effects. For example, cause 1 (C1) -> effect 1 (E1), but since effect 1 is also cause of effect 2, you should do cause 1 (C1) -> effect 1 (E1, C2) -> effect 2 (E2). 

        Example 6: (E2) The risk of insufficient control capacities (E2) is considered high in Zambia due to (C1) concurrent public health emergencies in the country (cholera, measles, COVID-19) (C1) that limit the country’s human and (E1, C2) financial capacities to respond to the current anthrax outbreak adequately (E1, C2).

        Example 7: (C1) Surveillance systems specifically targeting endemic transmission of chikungunya or Zika are weak or non-existent (C1) -> (E1, C2) Misdiagnosis between diseases  & Skewed surveillance (E1, C2) -> (E2, C3) Misinform policy decisions (E2, C3) -> (E3)reduced accuracy on the estimation of the true burden of each diseases (E3), poor risk assessments (E3), and non optimal clinical management and resource allocation (E3). 

        Example 8: (C1) Changes in the predominant circulating serotype (C1) -> (E1, C2) increase the population risk of subsequent exposure to a heterologous DENV serotype (E1, C2), -> (E2) which increases the risk of higher rates of severe dengue and deaths (E2).

        Now, extract the cause and effect from the following sentence:
        Text: "{text}"
        Cause:
        """
        response = self.predict(text=prompt)
        if response and hasattr(response, 'cause') and hasattr(response, 'effect'):
            return response.cause, response.effect
        else:
            return None

# Initialize the extraction module
extractor = CauseEffectExtractionModule()

# Function to apply the extractor to each row of the dataset
def extract_from_row(row):
    text = row['Text']  # Adjust the column name if necessary
    return extractor.forward(text)

# Apply the extraction function to the first 10 rows and store the results in a new column
who_data_assessment['CauseEffectPairs'] = who_data_assessment.head(20).apply(extract_from_row, axis=1)

st.title("WHO Data Assessment: Causality Extraction")
for index, row in who_data_assessment.head(20).iterrows():
    st.write(f"Text: {row['Text']}")
    st.write("Extracted Cause-Effect Pairs:")
    for pair in row['CauseEffectPairs']:
        if pair:
            cause, effect = pair
            st.write(f"- Cause: {cause}")
            st.write(f"- Effect: {effect}")
    st.write("---")
