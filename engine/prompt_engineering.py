import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm as tqdm
import pandas as pd
from utils.utils import create_folder_if_not_exist
from huggingface_hub import InferenceClient
import huggingface_hub
import os
import time


def create_folder(cfg):
    path = os.path.join(cfg.OUTPUT.BASE_PATH, cfg.MODEL.NAME, cfg.DATA.NAME, cfg.PROMPT.NAME)
    create_folder_if_not_exist(path)
    return path


class PromptEngineering:
    def __init__(self, model_name, cache_dir):
        self.model_name = model_name
        self.cache_dir = cache_dir

    def run(self, cfg, dataset):
        pass


class DeepseekCoder13b(PromptEngineering):
    def __init__(self, model_name, cache_dir):
        super().__init__(model_name, cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir
        )

    def run(self, cfg, dataset):
        path = create_folder(cfg)

        dataset_test = dataset.dataset_test
        outputs = []
        for ele in tqdm.tqdm(dataset_test):
            try:
                torch.cuda.empty_cache()
                xx = self.tokenizer.encode(ele['prompt_input'], return_tensors="pt").to(
                    self.model.device)
                output = self.model.generate(xx, max_new_tokens=15)

                output = self.tokenizer.decode(output[0][len(xx[0]):], skip_special_tokens=True)
            except torch.cuda.OutOfMemoryError:
                output = 'Error: OUT OF MEMORY'
            outputs.append(output)

        df = pd.DataFrame(dataset_test)  # Convert to DataFrame. Implementation depends on `Dataset`
        df['output'] = outputs  # pandas allows this operation

        df.to_csv(os.path.join(path, 'output.csv'), index=False)


class CodeLlama34b(PromptEngineering):
    def __init__(self, model_name, cache_dir):
        super().__init__(model_name, cache_dir)
        self.client = InferenceClient(model=self.model_name,
                                      token="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")

    def run(self, cfg, dataset):
        path = create_folder(cfg)

        dataset_test = dataset.dataset_test
        outputs = []
        for ele in tqdm.tqdm(dataset_test):
            try:
                output = self.client.text_generation(ele['prompt_input'], max_new_tokens=25)
                outputs.append(output)
            except huggingface_hub.utils._errors.HfHubHTTPError:
                time.sleep(4000)
                output = self.client.text_generation(ele['prompt_input'], max_new_tokens=15)
                outputs.append(output)
            except huggingface_hub.errors.ValidationError:
                output = 'Error: OUT OF MEMORY'
                outputs.append(output)

        df = pd.DataFrame(dataset_test)  # Convert to DataFrame. Implementation depends on `Dataset`
        df['output'] = outputs  # pandas allows this operation

        df.to_csv(os.path.join(path, 'output.csv'), index=False)


class CodeLlama7b(PromptEngineering):
    def __init__(self, model_name, cache_dir):
        super().__init__(model_name, cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir
        )
    def run(self, cfg, dataset):
        path = create_folder(cfg)
        self.model.to(cfg.MODEL.DEVICE)
        dataset_test = dataset.dataset_test
        outputs = []
        for ele in tqdm.tqdm(dataset_test):
            try:
                torch.cuda.empty_cache()
                xx = self.tokenizer.encode(ele['prompt_input'], return_tensors="pt").to(
                    self.model.device)
                output = self.model.generate(xx, max_new_tokens=155)
                # output = self.tokenizer.decode(output[0][len(xx[0]):], skip_special_tokens=True)
                output = self.tokenizer.decode(output[0], skip_special_tokens=True)

            except torch.cuda.OutOfMemoryError:
                output = 'Error: OUT OF MEMORY'
            outputs.append(output)
        df = pd.DataFrame(dataset_test)  # Convert to DataFrame. Implementation depends on `Dataset`
        df['output'] = outputs  # pandas allows this operation

        df.to_csv(os.path.join(path, 'output.csv'), index=False)



__REGISTERED_MODULES__ = {'deepseek-ai/deepseek-coder-1.3b-instruct': DeepseekCoder13b,
                          'codellama/CodeLlama-34b-Instruct-hf': CodeLlama34b,
                          'codellama/CodeLlama-7b-Instruct-hf': CodeLlama7b,
                          'meta-llama/Meta-Llama-3-70B-Instruct': CodeLlama34b,
                          'meta-llama/Meta-Llama-3-8B-Instruct': CodeLlama34b}


def build_prompt_engineering(cfg):
    cache_dir = cfg.TASK.CACHE_DIR
    model_name = cfg.MODEL.NAME
    prompt_engineering = __REGISTERED_MODULES__[model_name](model_name, cache_dir)
    return prompt_engineering
