import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

pd.set_option('display.max_columns', None)  # to display all columns
pd.set_option('display.expand_frame_repr', False)  # to disable line wrapping
pd.set_option('display.max_colwidth', None)  # to display full content of each cell
path = ("/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/output/meta-llama/Meta-Llama-3-8B-Instruct/Reid996/GPTCloneBench/zero_shot/output.csv")

df = pd.read_csv(path)


def transfer_output(ele):
    if ele.__contains__('Yes') and ele.__contains__('No'):
        return -1
    elif ele.__contains__('No'):
        return 0

    elif ele.__contains__('Yes'):
        return 1
    else:
        return -1

df['output'] = df['output'].fillna('Unknown')
# remove
df = df[~df['output'].str.contains('Error: OUT OF MEMORY')]
df = df[~df['output'].str.contains('Unknown')]

df['output_label'] = df['output'].apply(transfer_output)


# df = df[df['output'] != -1]

# Calculate Precision, Recall, and F1-Score
# precision = precision_score(df['label'], df['output'])
# recall = recall_score(df['label'], df['output'])
# f1 = f1_score(df['label'], df['output'])
#
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1-Score: {f1}")
