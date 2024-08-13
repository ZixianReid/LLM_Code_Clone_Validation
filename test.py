from tqdm import tqdm

for i, ele in tqdm(enumerate(range(11, 15)), leave=False):
    print(i)
    print(ele)


text = """
###Instruction:
Please analyze the following two code snippets and determine if they are code clones. Respond 
with only ###RESULT###@@YES@@ for clones or ###RESULT###@@NO@@ if not. Provide no other output.

Code snippet 1: $code_1
Code snippet 2: $code_2
###Response:
"""

split_text = text.split("###Response:", 1)
text_before_response = f'{split_text[0]}###Response:'

print(text_before_response)

def get_text_before_response(text: str) -> str:
    split_text = text.split("###Response:")
    text_before_response = f'{split_text[0]}###Response:'
    return text_before_response


