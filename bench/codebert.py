model_name = 'microsoft/codebert-base'

dataset_name = 'Reid996/big_clone_bench'
MAX_LENGTH = 255
import os
from transformers import RobertaTokenizerFast, RobertaTokenizerFast, RobertaForSequenceClassification, RobertaConfig, \
    Trainer, TrainingArguments
from tokenizers.processors import TemplateProcessing
import torch
from torch.utils.data import Dataset
from datasets import Dataset, load_dataset
import os
from pathlib import Path
import numpy as np
import evaluate
import accelerate
from transformers import EarlyStoppingCallback, IntervalStrategy
import os
import pandas as pd

tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
MAX_LENGTH = 255


def tokenization(row):
    tokenized_inputs = tokenizer([row["func1"], row["func2"]], padding="max_length", truncation=True,
                                 return_tensors="pt",
                                 max_length=MAX_LENGTH)
    tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"].flatten()
    tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"].flatten()
    return tokenized_inputs


config = RobertaConfig.from_pretrained(model_name, num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_dataset(dataset_name)

dataset_test = dataset['test'].map(tokenization, batched=False)
dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

dataset_train = dataset['train'].map(tokenization, batched=False)
dataset_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

dataset_eval = dataset['validation'].map(tokenization, batched=False)
dataset_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

accuracy = evaluate.load("accuracy")
recall = evaluate.load("recall")
precision = evaluate.load("precision")
f1 = evaluate.load("f1")

model = RobertaForSequenceClassification.from_pretrained(model_name, config=config, trust_remote_code=True).to(device)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "precision": precision.compute(predictions=predictions, references=labels)["precision"],
            "recall": recall.compute(predictions=predictions, references=labels)["recall"],
            "f1": f1.compute(predictions=predictions, references=labels)["f1"]}


BATCH_SIZE = 8
STEPS = 32
training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=2e-5,             # Learning rate
    adam_epsilon=1e-8,              # Epsilon for Adam optimizer
    num_train_epochs= 15,             # Total number of training epochs
    logging_dir='./logs',           # Directory for storing logs
    logging_steps=STEPS,
    evaluation_strategy="steps",
    eval_steps=STEPS,
    output_dir ="./output",
    dataloader_pin_memory=True,
    dataloader_num_workers=4, # how many cpus to use to load the data while training
    do_eval=True,                 # Perform evaluation at the end of training
    save_strategy="steps",
    save_steps=STEPS,
    load_best_model_at_end = True,
    metric_for_best_model = 'f1',
    save_total_limit=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_train,# Evaluation dataset
    compute_metrics=compute_metrics,
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=12)],
)

trainer.train()
# calculate the scores of the returning/best model on the evaluation dataset
trainer.evaluate()


# store model to disk (same as best checkpoint)
trainer.save_model(f"./fine_tuned_codebert")


trainer.evaluate(dataset_test)

