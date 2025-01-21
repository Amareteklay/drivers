import streamlit as st
import pandas as pd

# Title of the app
st.title("Causality Analysis Dashboard")

# Add a subtitle
st.subheader("Displaying Results")

# Create a sample DataFrame
data = {
    'Cause': ['High mobility', 'Limited access to care'],
    'Effect': ['Rapid spread of disease', 'Delayed treatment'],
    'Causality Type': ['Type 1', 'Type 2']
}
df = pd.DataFrame(data)

# Display the DataFrame in the app
st.write("### DataFrame of Causes and Effects:")
st.dataframe(df)

# Add user input for filtering
filter_text = st.text_input("Filter by Cause or Effect", "")
if filter_text:
    filtered_df = df[df.apply(lambda row: filter_text.lower() in row.to_string().lower(), axis=1)]
    st.write("### Filtered Results:")
    st.dataframe(filtered_df)
else:
    st.write("Enter text to filter the table.")

# Add a chart (optional)
st.write("### Bar Chart of Causality Types:")
st.bar_chart(df['Causality Type'].value_counts())
