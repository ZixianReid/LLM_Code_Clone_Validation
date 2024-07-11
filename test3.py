from datasets import load_dataset


dataset = load_dataset('Reid996/GPTCloneBench')

dataset = dataset['test']

xx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

filtered_dataset = [ele for ele in dataset if ele['id'] not in xx]
