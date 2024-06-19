import pandas as pd
from datasets import load_dataset

ds = load_dataset('google/code_x_glue_cc_clone_detection_big_clone_bench')

df_train = pd.DataFrame(ds['train'])

df_test = pd.DataFrame(ds['test'])

df_validation = pd.DataFrame(ds['validation'])

df_train['label'] = df_train['label'].apply(lambda x: 1 if x is True else 0)

df_test['label'] = df_test['label'].apply(lambda x: 1 if x is True else 0)

df_validation['label'] = df_validation['label'].apply(lambda x: 1 if x is True else 0)


df_train.to_csv('train.csv', index=False)

df_test.to_csv('test.csv', index=False)
df_validation.to_csv('validation.csv', index=False)

