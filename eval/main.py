import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

pd.set_option('display.max_columns', None)  # to display all columns
pd.set_option('display.expand_frame_repr', False)  # to disable line wrapping
pd.set_option('display.max_colwidth', None)  # to display full content of each cell
path = ("/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/output/meta-llama/Meta-Llama-3-8B-Instruct/Reid996/GPTCloneBench/zero_shot/output.csv")

df = pd.read_csv(path)


def transfer_output(ele):
    if ele.__contains__('###RESULT###@@YES@@'):
        return 1
    elif ele.__contains__('###RESULT###@@NO@@'):
        return 0
    else:
        return -1


# remove
# df = df[~df['output'].str.contains('Error: OUT OF MEMORY')]

df['output_label'] = df['output'].apply(transfer_output)

df = df[df['output_label'] == -1]

# df = df[df['output'] != -1]
#
# Calculate Precision, Recall, and F1-Score
# precision = precision_score(df['label'], df['output_label'])
# recall = recall_score(df['label'], df['output_label'])
# f1 = f1_score(df['label'], df['output_label'])
#
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1-Score: {f1}")
