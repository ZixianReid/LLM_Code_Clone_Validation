import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm as tqdm
import pandas as pd
from utils.utils import create_folder_if_not_exist
from huggingface_hub import InferenceClient
import huggingface_hub
import os
import time
from gradio_client import Client


def create_folder(cfg):
    path = os.path.join(cfg.OUTPUT.BASE_PATH, cfg.DATA.NAME, cfg.PROMPT.NAME)
    create_folder_if_not_exist(path)
    return path


class PromptEngineering:
    def __init__(self, model_name, cache_dir, device_map):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device_map = device_map

    def run(self, cfg, dataset):
        pass


class APIPromptEngineering(PromptEngineering):
    def __init__(self, model_name, cache_dir, device_map):
        super().__init__(model_name, cache_dir, device_map)
        self.client = InferenceClient(model=self.model_name,
                                      token="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")

    def run(self, cfg, dataset):
        path = create_folder(cfg)

        dataset_test = dataset.dataset_test
        outputs = []
        for ele in tqdm.tqdm(dataset_test):
            success = False
            while not success:
                try:
                    output = self.client.text_generation(ele['prompt_input'], max_new_tokens=15)
                    success = True
                except huggingface_hub.utils._errors.HfHubHTTPError:
                    time.sleep(300)
                    print("---------------------------")
                    print("meeting rate limits")
                except huggingface_hub.inference._text_generation.ValidationError or huggingface_hub.errors.ValidationError:
                    output = 'Error: OUT OF MEMORY'
                    success = True
                except Exception as e:
                    output = 'Error'
                    with open(os.path.join(path, 'errors.txt'), 'a') as f:
                        f.write(str(e))
                        f.write(' ')
                        f.write(str(ele['id']))
                        f.write('\n')
                    success = True
            outputs.append(output)
        df = pd.DataFrame(dataset_test)
        df['output'] = outputs

        df.to_csv(os.path.join(path, 'output.csv'), index=False)


class LocalMachinePromptEngineering(PromptEngineering):
    def __init__(self, model_name, cache_dir, device_map):
        super().__init__(model_name, cache_dir, device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
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
                # Encode the input text
                encoded_input = self.tokenizer(ele['prompt_input'], return_tensors="pt",
                                               padding=True)
                encoded_input = encoded_input.to(self.model.device)

                # Generate output using the correctly formatted arguments
                output = self.model.generate(
                    input_ids=encoded_input['input_ids'],
                    attention_mask=encoded_input['attention_mask'],
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=15,
                )

                # Decode generated tokens to text
                output = self.tokenizer.decode(output[0][len(encoded_input[0]):], skip_special_tokens=True)
            except torch.cuda.OutOfMemoryError:
                output = 'Error: OUT OF MEMORY'
            outputs.append(output)
        df = pd.DataFrame(dataset_test)  # Convert to DataFrame. Implementation depends on `Dataset`
        df['output'] = outputs  # pandas allows this operation

        df.to_csv(os.path.join(path, 'output.csv'), index=False)


class GradioPromptEngineering(PromptEngineering):
    def __init__(self, model_name, cache_dir, device_map):
        super().__init__(model_name, cache_dir, device_map)
        self.client = Client("Reid996/LLM_Code_Clone_Space")

    def run(self, cfg, dataset):
        path = create_folder(cfg)

        dataset_test = dataset.dataset_test
        outputs = []
        for ele in tqdm.tqdm(dataset_test):
            try:
                output = self.client.predict(
                    prompt=ele['prompt_input'],
                    api_name="/predict"
                )
            except Exception as e:
                print(e)
                output = 'Error: OUT OF MEMORY'
            outputs.append(output)

        df = pd.DataFrame(dataset_test)
        df['output'] = outputs

        df.to_csv(os.path.join(path, 'output1.csv'), index=False)

__REGISTERED_MODULES__ = {'deepseek-ai/deepseek-coder-1.3b-instruct': LocalMachinePromptEngineering,
                          'codellama/CodeLlama-34b-Instruct-hf': APIPromptEngineering,
                          'codellama/CodeLlama-7b-Instruct-hf': LocalMachinePromptEngineering,
                          'meta-llama/Meta-Llama-3-70B-Instruct': APIPromptEngineering,
                          'deepseek-ai/deepseek-coder-7b-instruct-v1.5': LocalMachinePromptEngineering,
                          'meta-llama/Meta-Llama-3-8B-Instruct': LocalMachinePromptEngineering,
                          "/home/zixian_z/PycharmProjects/LLM_Code_Clone_Validation/output_FT/deepseek-coder-1.3b-instruct": LocalMachinePromptEngineering,
                          "/home/zixian_z/PycharmProjects/LLM_Code_Clone_Validation/output_FT/checkpoint-2671": LocalMachinePromptEngineering}

__REGISTERED_DEVICE__ = {0: {"": 0}}


def build_prompt_engineering(cfg):
    cache_dir = cfg.TASK.CACHE_DIR
    model_name = cfg.MODEL.NAME
    device_map = __REGISTERED_DEVICE__[cfg.PROMPT.DEVICE_MAP]
    prompt_engineering = __REGISTERED_MODULES__[model_name](model_name, cache_dir, device_map)
    return prompt_engineering
