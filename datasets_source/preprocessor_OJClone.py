
import pickle
import pandas as pd


def read_pkl_file(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        if isinstance(data, pd.DataFrame):  # check if the read data is a DataFrame
            data.reset_index(drop=True, inplace=True)  # reset and drop the index column
    return data
#
#
# train_ids = '/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/OJClone/train/train_.pkl'
#
# test_ids = '/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/OJClone/test.txt/test_.pkl'
#
# valid_ids = '/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/OJClone/dev/dev_.pkl'
#
# programs_path = '/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/OJClone/programs.pkl'
#
# # data loading
# train_data = read_pkl_file(train_ids)
#
# test_data = read_pkl_file(test_ids)
#
# valid_data = read_pkl_file(valid_ids)
#
# programs = read_pkl_file(programs_path)
#
#
# def build_dataset(df, programs):
#     df['id'] = range(0, len(df))
#
#     df['fun1'] = df['id1'].map(lambda x: programs.loc[x, 1])
#
#     df['fun2'] = df['id2'].map(lambda x: programs.loc[x, 1])
#
#     return df
#
#
# train_df = build_dataset(train_data, programs)
# train_data = train_data[['id', 'id1', 'id2', 'fun1', 'fun2', 'label']]
#
# test_df = build_dataset(test_data, programs)
# test_data = test_data[['id', 'id1', 'id2', 'fun1', 'fun2', 'label']]
#
# valid_df = build_dataset(valid_data, programs)
# valid_data = valid_data[['id', 'id1', 'id2', 'fun1', 'fun2', 'label']]
#
# #export to csv order as [['id', 'id1', 'id2', 'fun1', 'fun2', 'label']]
# train_df.to_csv('/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/OJClone/train/train.csv', index=False)
# test_df.to_csv('/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/OJClone/test.txt/test.txt.csv', index=False)
# valid_df.to_csv('/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/OJClone/dev/validation.csv', index=False)