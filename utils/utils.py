import os
from symbol import import_name

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
    print("Fine Tuning Output Dir: {}".format(cfg.FINE_TUNING.OUTPUT_DIR))
    print("------------------------------------")
    print("Positive Flag: {}".format(cfg.PROMPT.POSITIVE_FLAG))
    print("------------------------------------")
    print("Negative Flag: {}".format(cfg.PROMPT.NEGATIVE_FLAG))
    print("------------------------------------")
    print("Fine Tuning Lora R: {}".format(cfg.FINE_TUNING.LORA_R))
    print("------------------------------------")
    print("Per Device Train Batch Size: {}".format(cfg.FINE_TUNING.PER_DEVICE_TRAIN_BATCH_SIZE))
    print("------------------------------------")
    print("Per Device Eval Batch Size: {}".format(cfg.FINE_TUNING.PER_DEVICE_EVAL_BATCH_SIZE))
    print("------------------------------------")
    print("Gradient Accumulation Steps: {}".format(cfg.FINE_TUNING.GRADIENT_ACCUMULATION_STEPS))
    print("------------------------------------")
    print("Fine tuning max length: {}".format(cfg.FINE_TUNING.MAX_SEQ_LENGTH))
    print("------------------------------------")


def wandb_configuration(cfg):
    import os
    import wandb
    wandb.login(key='4bea86af0721fc5785732fd0fd63fb8b8a9ca323')
    os.environ["WANDB_PROJECT"] = "LLM_code_clone_detection"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
