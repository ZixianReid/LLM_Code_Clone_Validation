import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

pd.set_option('display.max_columns', None)  # to display all columns
pd.set_option('display.expand_frame_repr', False)  # to disable line wrapping
pd.set_option('display.max_colwidth', None)  # to display full content of each cell
path = ("/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/output/deepseek-ai/deepseek-coder-1.3b-instruct/Reid996/OJClone_code_clone_unbalanced/few_shot/output.csv")

df = pd.read_csv(path)


def transfer_output(ele):
    if ele.__contains__('Yes'):
        return 1
    elif ele.__contains__('No'):
        return 0
    else:
        return -1


# remove
df = df[~df['output'].str.contains('Error: OUT OF MEMORY')]

df['output'] = df['output'].apply(transfer_output)



df['output'] = df.apply(lambda row: 0 if (row['output'] == -1 and row['label'] == 1) else (1 if (row['output'] == -1 and row['label'] == 0) else row['output']), axis=1)


# Calculate Precision, Recall, and F1-Score
precision = precision_score(df['label'], df['output'])
recall = recall_score(df['label'], df['output'])
f1 = f1_score(df['label'], df['output'])

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
