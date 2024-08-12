from accelerate.commands.config.config_args import cache_dir
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir='/data/zixian_z/huggingface')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir='/data/zixian_z/huggingface')


print(tokenizer.pad_token)
print(tokenizer.eos_token)
