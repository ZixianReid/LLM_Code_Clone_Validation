import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


class FineTuningEngineering:
    def __init__(self, model_name, cache_dir, bnb_config, peft_config, training_arguments):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.bnb_config = bnb_config
        self.peft_config = peft_config
        self.training_arguments = training_arguments

    def train(self, cfg, dataset):
        pass


class CodeLlama7b(FineTuningEngineering):
    def __init__(self, model_name, cache_dir, bnb_config, peft_config, training_arguments):
        super().__init__(model_name, cache_dir, bnb_config, peft_config, training_arguments)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config,
                                                          cache_dir=self.cache_dir, device_map="auto")
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
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset_train,
            peft_config=self.peft_config,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            packing=packing
        )

        trainer.train()

        trainer.model.save_pretrained(new_model)


__REGISTERED_MODULES__ = {'codellama/CodeLlama-7b-Instruct-hf': CodeLlama7b,
                          'deepseek-ai/deepseek-coder-1.3b-instruct': CodeLlama7b,
                          'deepseek-ai/deepseek-coder-7b-instruct-v1.5': CodeLlama7b}


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

    training_arguments = TrainingArguments(
        output_dir=output_dir,
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
        report_to="tensorboard",
        deepspeed= "/home/zixian_z/PycharmProjects/LLM_Code_Clone_Validation/config/ds/llama2_ds_zero3_config.json"
    )

    fine_tuning_model = __REGISTERED_MODULES__[model_name](model_name, cache_dir, bnb_config, peft_config,
                                                           training_arguments)

    return fine_tuning_model
