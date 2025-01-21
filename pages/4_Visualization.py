import streamlit as st
import pandas as pd
from src.visualization import create_bar_chart

st.title("Visualization")

data = pd.DataFrame({"Category": ["Transport", "Environment"], "Count": [10, 5]})
chart = create_bar_chart(data, "Category", "Count")
st.plotly_chart(chart)
