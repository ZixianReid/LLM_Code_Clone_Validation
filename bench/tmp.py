from datasets import load_dataset
import wandb
from tqdm import tqdm

dataset = load_dataset("yahoo_answers_topics")
label_list = dataset['train'].unique('topic')
label_list.sort()
num_labels = len(label_list)

dataset = dataset.rename_column('topic', 'labels')

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

dataset = dataset.map(lambda x: tokenizer(x['question_title'], truncation=True), batched=True)

from transformers import AutoModelForSequenceClassification
from transformers.integrations import WandbCallback
import torch
import pandas as pd

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

def get_topic(sentence, tokenize=tokenizer, model=model):
    # tokenize the input
    inputs = tokenizer(sentence, return_tensors='pt')
    # ensure model and inputs are on the same device (GPU)
    inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
    model = model.cuda()
    # get prediction - 10 classes "probabilities" (not really true because they still need to be normalized)
    with torch.no_grad():
        predictions = model(**inputs)[0].cpu().numpy()
    # get the top prediction class and convert it to its associated label
    top_prediction = predictions.argmax().item()
    return dataset['train'].features['labels'].int2str(top_prediction)

class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows to visualize the
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset
          for generating predictions.
        num_samples (int, optional): Number of samples to select from
          the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, trainer, tokenizer, val_dataset,
                 num_samples=100, freq=2):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated
              with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from
              the validation dataset for generating predictions.
              Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.model = trainer.model
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def get_topic(sentence, tokenize=tokenizer, model=model):
        # tokenize the input
        inputs = tokenizer(sentence, return_tensors='pt')
        # ensure model and inputs are on the same device (GPU)
        inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
        model = model.cuda()
        # get prediction - 10 classes "probabilities" (not really true because they still need to be normalized)
        with torch.no_grad():
            predictions = model(**inputs)[0].cpu().numpy()
        # get the top prediction class and convert it to its associated label
        top_prediction = predictions.argmax().item()
        return dataset['train'].features['labels'].int2str(top_prediction)


    def generate(self, sentence):
        inputs = tokenizer(sentence, return_tensors='pt')
        # ensure model and inputs are on the same device (GPU)
        inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
        model = self.model.cuda()
        # get prediction - 10 classes "probabilities" (not really true because they still need to be normalized)
        with torch.no_grad():
            predictions = model(**inputs)[0].cpu().numpy()
        # get the top prediction class and convert it to its associated label
        top_prediction = predictions.argmax().item()
        return dataset['train'].features['labels'].int2str(top_prediction)

    def samples_tables(self, examples):
        records_table = wandb.Table(columns=["prompt", "generation"])
        for example in tqdm(examples, leave=False):
            sentence = example['question_title']
            generation = self.generate(sentence)
            records_table.add_data(sentence, generation)

        return records_table

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)

        records_table = self.samples_tables(self.sample_dataset)
        self._wandb.log({"sample_predictions":records_table})



from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    report_to='wandb',
    run_name='reid',
    output_dir='topic_classification',  # output directory
    overwrite_output_dir=True,
    evaluation_strategy='steps',  # check evaluation metrics at each epoch
    learning_rate=5e-5,  # we can customize learning rate
    max_steps=30000,
    logging_steps=100,  # we will log every 100 steps
    eval_steps=5000,  # we will perform evaluation every 500 steps
    save_steps=10000,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'  # name of the W&B run
)

from datasets import load_metric
import numpy as np

accuracy_metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # metrics from the datasets library have a `compute` method
    return accuracy_metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,  # model to be trained
    args=args,  # training args
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,  # for padding batched data
    compute_metrics=compute_metrics  # for custom metrics
)

progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    val_dataset=dataset['test'],
    num_samples=10,
    freq=2,
)

trainer.add_callback(progress_callback)

trainer.train()
