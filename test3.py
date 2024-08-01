from accelerate.commands.config.config_args import cache_dir
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('/home/zixian_z/PycharmProjects/LLM_Code_Clone_Validation/output_FT/llama8b',
                                             cache_dir = '/data/zixian_z/huggingface')

