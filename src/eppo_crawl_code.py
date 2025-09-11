import sqlite3
import pandas as pd

# %pip install python-docx
# from docx import Document  # Import the python-docx library
# Connect to the SQLite database
connection = sqlite3.connect("data/eppo_downloads/eppocodes.sqlite")
# connection1 = sqlite3.connect("../data/eppocodes_all.sqlite")
# Define the query to retrieve all rows from the t_codes table
query = """--sql 
SELECT * FROM t_codes;
"""

# Use pandas to execute the query and load the data into a DataFrame
t_codes_df = pd.read_sql_query(query, connection)

# save to .csv
t_codes_df.to_csv("data/eppo_downloads/eppo_code.csv", index=False)
