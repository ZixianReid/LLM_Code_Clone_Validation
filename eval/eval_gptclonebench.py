import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sympy.assumptions.cnf import AND

pd.set_option('display.max_columns', None)  # to display all columns
pd.set_option('display.expand_frame_repr', False)  # to disable line wrapping
pd.set_option('display.max_colwidth', None)  # to display full content of each cell
path = (
    "/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/output/meta-llama/Meta-Llama-3-8B-Instruct/Reid996/GPTCloneBench/zero_shot/output.csv")

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
#
#
# def recall_for_each_clone_type(df):
#     unique_clone_types = df['clone_type'].unique()
#     recall_scores = {}
#
#     for clone_type in unique_clone_types:
#         temp_df = df[df['clone_type'] == clone_type]
#         recall = recall_score(temp_df['label'], temp_df['output_label'])
#         recall_scores[clone_type] = recall
#
#     for clone_type, recall in recall_scores.items():
#         print(f"Recall for clone_type {clone_type}: {recall}")
#
#
# recall_for_each_clone_type(df)
# from util import line_based_similarity
#
# df['similarity_score'] = df.apply(line_based_similarity, axis=1)
#
# df = df[(df['clone_type'] != 'T1-2')]
#
# import seaborn as sns
# from matplotlib import pyplot as plt
#
#
# def hist_chart_proportion(df):
#     # Set the style of Seaborn for prettier plots
#     sns.set_theme(style="whitegrid")
#
#     # Create figure and axis variables for 1x1 subplot
#     fig, axs = plt.subplots(figsize=(10, 10))
#
#     # Bin the data frame by similarity score with 20 bins
#     df['binned'] = pd.cut(df['similarity_score'], bins=20)
#
#     # Calculate the proportion of false positives within each bin
#     proportion_df = df.groupby('binned')['output_label'].value_counts(normalize=True).unstack().fillna(0)
#     proportion_df.columns = ['False Positive', 'True Positive']
#
#     # Reset index to make 'binned' a column
#     proportion_df.reset_index(inplace=True)
#
#     # Plotting the proportion of false positives for each bin
#     sns.barplot(data=proportion_df, x='binned', y='False Positive', color='red', ax=axs)
#
#     axs.set_title('Proportion of False Positives per Bin')
#     axs.set_xlabel('Similarity Score Bins')
#     axs.set_ylabel('Proportion of False Positives')
#     axs.set_xticklabels(axs.get_xticklabels(), rotation=45)
#     axs.grid(False)
#
#     plt.tight_layout()
#     plt.show()
#
# hist_chart_proportion(df)