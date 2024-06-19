from datasets import load_dataset
import pandas as pd
pd.set_option('display.max_columns', None)  # to display all columns
pd.set_option('display.expand_frame_repr', False)  # to disable line wrapping
pd.set_option('display.max_colwidth', None)  # to display full content of each cell

# Load the dataset
dataset = load_dataset('Reid996/big_clone_bench')

# Convert the dataset to pandas
dataset = dataset['test'].to_pandas()

# Append func1 with itself and place it into the new column 'tmp'
dataset['tmp'] = dataset['func1'] + dataset['func1']

# Calculate the length of the strings in the 'tmp' column.
dataset['length_tmp'] = dataset['tmp'].str.len()

# Sort the values by the 'length_tmp' column.
dataset = dataset.sort_values(by='length_tmp', ascending=False)

for row in dataset.head(10)['tmp']:
    print(row)
    print("------------------------------------------")


