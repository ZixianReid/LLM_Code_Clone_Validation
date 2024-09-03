import pandas as pd


xx = pd.read_csv('/home/zixian_z/PycharmProjects/LLM_Code_Clone_Validation/output/Reid996/big_clone_bench/zero_shot/output.csv')

zz = xx['output']

yy = xx['label']
print(zz.head(10))
print(yy.head(10))