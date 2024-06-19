import pandas as pd
import os


train_path = '/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/BCB/traindata.txt'

test_path = '/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/BCB/testdata.txt'

validation_path = '/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/BCB/devdata.txt'


def read_txt_to_df(file_path):
    """Read a text file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path, sep='\t', names=['path1', 'path2', 'label'])
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


train_df = read_txt_to_df(train_path)

test_df = read_txt_to_df(test_path)

validation_df = read_txt_to_df(validation_path)


def load_file(file_path):
    base_path = '/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/BCB'
    path_1 = os.path.join(base_path, file_path)
    with open(path_1, 'r') as file:
        text = file.read()
    return text
def build(df):
    base_path = '/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/BCB'
    df['label'] = df['label'].apply(lambda x: 1 if x == 1 else 0)
    df['id1'] = df['path1'].apply(lambda x: os.path.basename(x).split('.txt')[0])
    df['id2'] = df['path2'].apply(lambda x: os.path.basename(x).split('.txt')[0])
    df['fun1'] = df['path1'].apply(load_file)
    df['fun2'] = df['path2'].apply(load_file)
    df['id'] = range(0, len(df))
    df = df.drop(['path1', 'path2'], axis=1)
    return df

train_df = build(train_df)

test_df = build(test_df)

validation_df = build(validation_df)
def save_df(df, file_path, columns_order):
    """
    Save a pandas DataFrame to a file, with a specific column order.
    df: DataFrame to save
    file_path: Path to the file to save the DataFrame
    columns_order: A list of column names in the order you want them
    """
    try:
        df = df[columns_order]
        df.to_csv(file_path, index=False)
        print(f"DataFrame saved to: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

columns_order = ['id', 'id1', 'id2', 'fun1', 'fun2', 'label']


train_out_path = '/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/BCB/train.csv'

test_out_path = '/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/BCB/test.csv'

validation_out_path = '/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/BCB/validation.csv'

save_df(train_df, train_out_path, columns_order)

save_df(test_df, test_out_path, columns_order)

save_df(validation_df, validation_out_path, columns_order)