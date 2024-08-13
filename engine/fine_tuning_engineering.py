import os
import torch
from datasets import load_dataset
from numpy.lib.function_base import select
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from tqdm import tqdm
import wandb
from transformers.integrations import WandbCallback
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


class FineTuningEngineering:
    def __init__(self, model_name, cache_dir, bnb_config, peft_config, training_arguments):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.bnb_config = bnb_config
        self.peft_config = peft_config
        self.training_arguments = training_arguments

    def train(self, cfg, dataset):
        pass


class RemoteMachineFineTuning(FineTuningEngineering):
    def __init__(self, model_name, cache_dir, bnb_config, peft_config, training_arguments):
        super().__init__(model_name, cache_dir, bnb_config, peft_config, training_arguments)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config,
                                                          cache_dir=self.cache_dir)
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    def train(self, cfg, dataset):
        max_seq_length = cfg.FINE_TUNING.MAX_SEQ_LENGTH
        packing = cfg.FINE_TUNING.PACKING
        new_model = cfg.FINE_TUNING.NEW_MODEL_NAME
        dataset_train = dataset.dataset_train
        dataset_val = dataset.dataset_val

        class SkipEvaluationTrainer(SFTTrainer):
            def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
                # Return an empty dictionary or minimal metrics to mimic an evaluation without computation
                return {f"{metric_key_prefix}_loss": 0.0}  # Ex

        trainer = SkipEvaluationTrainer(
            model=self.model,
            train_dataset=dataset_train,
            eval_dataset=dataset_val,
            peft_config=self.peft_config,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            packing=packing,
            data_collator=DataCollatorForCompletionOnlyLM(instruction_template=self.tokenizer.encode(cfg.PROMPT.INSTRUCTION_TEMPLATE, add_special_tokens=False)[2:],
                                                          response_template=self.tokenizer.encode(cfg.PROMPT.RESPONSE_TEMPLATE, add_special_tokens=False)[2:],
                                                          tokenizer=self.tokenizer)
        )

        progress_callback = WandbPredictionProgressCallback(
            trainer=trainer,
            tokenizer=self.tokenizer,
            val_dataset=dataset_val,
            num_samples=10
        )

        trainer.add_callback(progress_callback)
        trainer.train()
        trainer.save_model(new_model)


__REGISTERED_MODULES__ = {'codellama/CodeLlama-7b-Instruct-hf': RemoteMachineFineTuning,
                          'deepseek-ai/deepseek-coder-1.3b-instruct': RemoteMachineFineTuning,
                          'deepseek-ai/deepseek-coder-7b-instruct-v1.5': RemoteMachineFineTuning,
                          'meta-llama/Meta-Llama-3-8B-Instruct': RemoteMachineFineTuning}


def build_fine_tuning_model(cfg):
    model_name = cfg.MODEL.NAME
    cache_dir = cfg.TASK.CACHE_DIR
    bnb_4bit_compute_dtype = cfg.FINE_TUNING.BNB_4BIT_COMPUTE_DTYPE
    use_4bit = cfg.FINE_TUNING.USE_4BIT
    bnb_4bit_quant_type = cfg.FINE_TUNING.BNB_4BIT_QUANT_TYPE
    use_nested_quant = cfg.FINE_TUNING.USE_NESTED_QUANT
    lora_alpha = cfg.FINE_TUNING.LOAR_ALPHA
    lora_dropout = cfg.FINE_TUNING.LOAR_DROPOUT
    lora_r = cfg.FINE_TUNING.LORA_R
    output_dir = cfg.FINE_TUNING.OUTPUT_DIR
    num_train_epochs = cfg.FINE_TUNING.NUM_TRAIN_EPOCHS
    per_device_train_batch_size = cfg.FINE_TUNING.PER_DEVICE_TRAIN_BATCH_SIZE
    gradient_accumulation_steps = cfg.FINE_TUNING.GRADIENT_ACCUMULATION_STEPS
    optim = cfg.FINE_TUNING.OPTIM
    save_steps = cfg.FINE_TUNING.SAVE_STEPS
    logging_steps = cfg.FINE_TUNING.LOGGING_STEPS
    learning_rate = cfg.FINE_TUNING.LEARNING_RATE
    weight_decay = cfg.FINE_TUNING.WEIGHT_DECAY
    fp16 = cfg.FINE_TUNING.FP16
    bf16 = cfg.FINE_TUNING.BF16
    max_grad_norm = cfg.FINE_TUNING.MAX_GRAD_NORM
    max_steps = cfg.FINE_TUNING.MAX_STEPS
    warmup_ratio = cfg.FINE_TUNING.WARMUP_RATIO
    group_by_length = cfg.FINE_TUNING.GROUP_BY_LENGTH
    lr_scheduler_type = cfg.FINE_TUNING.LR_SCHEDULER_TYPE

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    if compute_dtype == torch.bfloat16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # collator = DataCollatorForCompletionOnlyLM(instruction_template="<s>[INST]",
    #                                            response_template=response_template, tokenizer=tokenizer, mlm=False)
    wandb.init(dir='/data/zixian_z/')
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        report_to='wandb',
        run_name=cfg.MODEL.NAME,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy='epoch',
        deepspeed="/home/zixian_z/PycharmProjects/LLM_Code_Clone_Validation/config/ds/llama2_ds_zero3_config.json"
    )

    fine_tuning_model = __REGISTERED_MODULES__[model_name](model_name, cache_dir, bnb_config, peft_config,
                                                           training_arguments)

    return fine_tuning_model


class WandbPredictionProgressCallback(WandbCallback):
    def __init__(self, trainer, tokenizer, val_dataset,
                 num_samples=10):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
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


    def generate(self, ele):
        def get_text_before_response(text: str) -> str:
            split_text = text.split("###Response:")
            text_before_response = f'{split_text[0]}###Response:'
            return text_before_response
        text = get_text_before_response(ele['text'])

        encoded_input = self.tokenizer(ele['prompt_input'], return_tensors="pt",
                                               padding=True)
        output = self.model.generate(
            input_ids=encoded_input['input_ids'],
            attention_mask=encoded_input['attention_mask'],
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=15,
        )
        output = self.tokenizer.decode(output[0][len(encoded_input[0]):], skip_special_tokens=True)
        return output


    def samples_tables(self, examples):
        records_table = wandb.Table(columns=["num", "generation"])
        for i, ele in tqdm(enumerate(examples), leave=False):
            generation = self.generate(ele)
            records_table.add_data(i, generation)

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)

        records_table = self.samples_tables(self.sample_dataset)
        self._wandb.log({"sample_predictions": records_table})

        return records_table