import os

base_path = '/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/datasets_source/BCB'

path_1 = './bigclonebenchdata/17156131.txt'

path_2 = os.path.join(base_path, path_1)

#read text from path_2
with open(path_2, 'r') as file:
    text = file.read()

print(text)