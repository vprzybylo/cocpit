import pandas as pd
import sqlite3

# Read sqlite query results into a pandas DataFrame
#con = sqlite3.connect("saved_models/2000_ARM.db")
#df = pd.read_sql_query("SELECT * FROM 2000_ARM", con)
df = pd.read_sql_table('2000_ARM', 'sqlite:///../saved_models/2000_ARM.db')  

# Verify that result of SQL query is stored in the dataframe
print(df.head)

