
from huggingface_hub import InferenceClient


client = InferenceClient(model="codellama/CodeLlama-34b-Instruct-hf", token="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")

output = client.text_generation("write hello world in python")
print(output)