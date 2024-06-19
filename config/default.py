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

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'deepseek-ai/deepseek-coder-1.3b-instruct'

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
__BASE_PATH = '/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/output'
_C.OUTPUT = CN()
_C.OUTPUT.PATH = os.path.join(__BASE_PATH, _C.DATA.NAME, _C.PROMPT.NAME)


if __name__ == '__main__':
    def create_folder_if_not_exist(path):
        if not os.path.exists(path):
            os.makedirs(path)

    print(_C.OUTPUT.PATH)
    create_folder_if_not_exist(_C.OUTPUT.PATH)