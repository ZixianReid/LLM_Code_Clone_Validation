#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

python /home/zixian_z/PycharmProjects/LLM_Code_Clone_Validation/main.py --config_file /home/zixian_z/PycharmProjects/LLM_Code_Clone_Validation/config/human_and_gpt/ft/llama8b_gptclonebench.yaml


python /home/zixian_z/PycharmProjects/LLM_Code_Clone_Validation/main.py --config_file /home/zixian_z/PycharmProjects/LLM_Code_Clone_Validation/config/human_and_gpt/ft/llama8b_bcb.yaml
