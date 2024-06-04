import pandas as pd
from sqlalchemy import create_engine

# Database connection details
username = 'aiotmrdev'
password = 'Rmkm11!!88**'
hostname = 'localhost'
port = '3306'
database_name = 'pub'

# Connect to MySQL database
engine = create_engine(f'mysql+mysqlconnector://{username}:{password}@{hostname}:{port}/{database_name}')

# Define file paths
links_csv_file = 'links.csv'
books_csv_file = 'books.csv'

# Load CSV data into DataFrames
links_df = pd.read_csv(links_csv_file)
books_df = pd.read_csv(books_csv_file)

# Write the DataFrames to the database
links_df.to_sql('links', engine, if_exists='replace', index=False)
books_df.to_sql('books', engine, if_exists='replace', index=False)

