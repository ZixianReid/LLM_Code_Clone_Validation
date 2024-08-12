from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode some input text
input_text = "The science of today is the technology of tomorrow."
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text with a specific max_length
output = model.generate(input_ids, max_length=50)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)



