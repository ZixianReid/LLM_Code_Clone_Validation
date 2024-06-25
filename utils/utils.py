import os
from transformers import AutoTokenizer


def create_folder_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_token_length(model_name: str, text: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Encode the input text
    encoded_input = tokenizer.encode(text, return_tensors='pt')

    # Return the length of the encoded input.
    return len(encoded_input[0])


def print_info(cfg):
    print("Configuration:Done")
    print("-----------------------------------")
    print("Execution Task: {}".format(cfg.TASK.NAME))
    print("------------------------------------")
    print("Model Name: {}".format(cfg.MODEL.NAME))
    print("------------------------------------")
    print("Dataset Name: {}".format(cfg.DATA.NAME))
    print("------------------------------------")
    print("Prompt Style: {}".format(cfg.PROMPT.NAME))
    print("------------------------------------")
