# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaTokenizerFast, RobertaTokenizerFast, RobertaForSequenceClassification, RobertaConfig
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = RobertaConfig.from_pretrained('4luc/codebert-code-clone-detector', num_labels=2)
model = RobertaForSequenceClassification.from_pretrained("4luc/codebert-code-clone-detector",
                                                         config=config, trust_remote_code=True).to(device)
dataset_name = 'Reid996/GPTCloneBench'
MAX_LENGTH = 255

model_name = 'microsoft/codebert-base'
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

dataset = load_dataset(dataset_name)

def tokenization(row):
    tokenized_inputs = tokenizer([row["func1"], row["func2"]], padding="max_length", truncation=True,
                                 return_tensors="pt", max_length=MAX_LENGTH)
    tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"].flatten()
    tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"].flatten()
    return tokenized_inputs

dataset_test = dataset['test']
# dataset_test = dataset_test.filter(lambda example: example['clone_type'] == 'MT3')


# Load dataset and prepare data loader for inference
# test_loader = DataLoader(dataset_test, batch_size=16, shuffle=False)

#
# Define the function to perform inference on the test dataset

def evaluate(model, dataset):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            tokenized_inputs = tokenization(sample).to(device)
            input_ids = tokenized_inputs['input_ids'].unsqueeze(0).to(device)
            attention_mask = tokenized_inputs['attention_mask'].unsqueeze(0).to(device)
            label = sample['label']

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1)

            predictions.append(
                pred.cpu().numpy())  # Changed to numpy array as precision_recall_fscore_support needs an array-like object
            true_labels.append(label)

            # Calculate precision, recall, and f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    return precision, recall, f1


# Call the evaluate function
precision, recall, f1 = evaluate(model, dataset_test)

# Output the results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


input = tokenizer('text', return_tensors="pt", truncation=True, padding=True).to(device)

outputs = model.forward(input['input_ids'], attention_mask=input['attention_mask'])

print(outputs)


