import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

pd.set_option('display.max_columns', None)  # to display all columns
pd.set_option('display.expand_frame_repr', False)  # to disable line wrapping
pd.set_option('display.max_colwidth', None)  # to display full content of each cell
path = (
    "/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/output/meta-llama/Meta-Llama-3-70B-Instruct/Reid996/big_clone_bench/zero_shot/output.csv")

df = pd.read_csv(path)


def preprocess(df):
    def transfer_output(ele):
        if ele.__contains__('###RESULT###@@YES@@'):
            return 1
        elif ele.__contains__('###RESULT###@@NO@@'):
            return 0
        else:
            return -1

    df['output_label'] = df['output'].apply(transfer_output)
    #
    df = df[df['output_label'] != -1]

    return df


df = preprocess(df)


#
def recall_precision_f1(df):
    precision = precision_score(df['label'], df['output_label'])
    recall = recall_score(df['label'], df['output_label'])
    f1 = f1_score(df['label'], df['output_label'])

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")


recall_precision_f1(df)


def recall_for_each_clone_type(df):
    unique_clone_types = df['clone_type'].unique()
    recall_scores = {}

    for clone_type in unique_clone_types:
        temp_df = df[df['clone_type'] == clone_type]
        recall = recall_score(temp_df['label'], temp_df['output_label'])
        recall_scores[clone_type] = recall

    for clone_type, recall in recall_scores.items():
        print(f"Recall for clone_type {clone_type}: {recall}")


recall_for_each_clone_type(df)
