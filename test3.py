import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "meta-llama/Meta-Llama-3-8B"


tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)


encoded_input = tokenizer("hello", return_tensors="pt",
                                               padding=True)


                # Generate output using the correctly formatted arguments
output = model.generate(
                input_ids=encoded_input['input_ids'],
                    attention_mask=encoded_input['attention_mask'],
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=15
                )

output = tokenizer.decode(output[0][len(encoded_input[0]):], skip_special_tokens=True)