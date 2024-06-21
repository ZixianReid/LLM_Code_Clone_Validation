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
# Output
# -----------------------------------------------------------------------------
_C.OUTPUT = CN()
# _C.OUTPUT.PATH = os.path.join(__BASE_PATH, _C.DATA.NAME, _C.PROMPT.NAME)

_C.OUTPUT.BASE_PATH = '/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/output'

if __name__ == '__main__':
    print(_C.TASK.CACHE_DIR)
