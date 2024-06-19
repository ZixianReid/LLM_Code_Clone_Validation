import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm as tqdm
import pandas as pd
from utils.utils import create_folder_if_not_exist
import os

class PromptEngineering:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    def run(self, cfg, dataset):
        dataset_test = dataset.dataset_test
        outputs = []
        for ele in tqdm.tqdm(dataset_test):
            try:
                torch.cuda.empty_cache()
                xx = self.tokenizer.encode(ele['prompt_input'], return_tensors="pt").to(
                self.model.device)
                output = self.model.generate(xx, max_new_tokens=5)
                output = self.tokenizer.decode(output[0][len(xx[0]):], skip_special_tokens=True)
            except torch.cuda.OutOfMemoryError:
                output = 'Error: OUT OF MEMORY'
            outputs.append(output)

        df = pd.DataFrame(dataset_test)  # Convert to DataFrame. Implementation depends on `Dataset`
        df['output'] = outputs  # pandas allows this operation

        create_folder_if_not_exist(cfg.OUTPUT.PATH)
        df.to_csv(os.path.join(cfg.OUTPUT.PATH, 'output.csv'), index=False)


def build_prompt_engineering(cfg):
    model_name = cfg.MODEL.NAME
    prompt_engineering = PromptEngineering(model_name)
    return prompt_engineering
