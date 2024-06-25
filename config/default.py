import os

from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Task definition
# -----------------------------------------------------------------------------
_C.TASK = CN()
_C.TASK.NAME = "prompt_engineering"  # fine_tuning and prompt_engineering
_C.TASK.CACHE_DIR = None

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'deepseek-ai/deepseek-coder-1.3b-instruct'
_C.MODEL.DEVICE = 'cuda:0'

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.NAME = 'Reid996/OJClone_code_clone_unbalanced'

# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------
_C.PROMPT = CN()
_C.PROMPT.NAME = 'zero_shot'
_C.PROMPT.INDICATE_TRUE = "Yes"
_C.PROMPT.INDICATE_FALSE = "No"

# -----------------------------------------------------------------------------
# Fine-tuning
# -----------------------------------------------------------------------------
_C.FINE_TUNING = CN()
_C.FINE_TUNING.NEW_MODEL_NAME = 'codellama2-code_clone'

_C.FINE_TUNING.LOAR_R = 64
_C.FINE_TUNING.LOAR_ALPHA = 16
_C.FINE_TUNING.LOAR_DROPOUT = 0.1



_C.FINE_TUNING.USE_4BIT = True
_C.FINE_TUNING.BNB_4BIT_COMPUTE_DTYPE = "float16"
_C.FINE_TUNING.BNB_4BIT_QUANT_TYPE = "nf4"
_C.FINE_TUNING.USE_NESTED_QUANT=False



_C.FINE_TUNING.NUM_TRAIN_EPOCHS = 10
_C.FINE_TUNING.FP16 = False
_C.FINE_TUNING.BF16 = False


_C.FINE_TUNING.PER_DEVICE_TRAIN_BATCH_SIZE = 4
_C.FINE_TUNING.PER_DEVICE_EVAL_BATCH_SIZE = 4
_C.FINE_TUNING.GRADIENT_ACCUMULATION_STEPS = 1
_C.FINE_TUNING.GRADIENT_CHECKPOINT = True

_C.FINE_TUNING.MAX_GRAD_NORM = 0.3
_C.FINE_TUNING.LEARNING_RATE = 2e-4
_C.FINE_TUNING.WEIGHT_DECAY = 0.001
_C.FINE_TUNING.OPTIM = 'paged_adamw_32bit'
_C.FINE_TUNING.LR_SCHEDULER_TYPE = 'cosine'
_C.FINE_TUNING.WARMUP_RATIO = 0.03

_C.FINE_TUNING.GROUP_BY_LENGHT = True
_C.FINE_TUNING.SAVE_STEPS = 0
_C.FINE_TUNING.LOGGING_STEPS = 25

_C.FINE_TUNING.MAX_SEQ_LENGTH = None
_C.FINE_TUNING.PACKING = False
_C.FINE_TUNING.DEVICE_MAP = 'cuda:0'




# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
_C.OUTPUT = CN()
# _C.OUTPUT.PATH = os.path.join(__BASE_PATH, _C.DATA.NAME, _C.PROMPT.NAME)

_C.OUTPUT.BASE_PATH = '/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/output'

if __name__ == '__main__':
    print(_C.TASK.CACHE_DIR)
