from string import Template

prefix_ft = "<s>[INST]"

Simple_template = """
Please analyze the following two code snippets and determine if they are code clones. Respond 
with only ###RESULT###@@YES@@ for clones or ###RESULT###@@NO@@ if not. Provide no other output.
"""


similarity_template = """
Compare the following two code snippets and rate their similarity on a scale from 0 to 10, 
where 0 indicates no similarity and 10 indicates maximum similarity. Respond with only the similarity score as your 
output."""

input_PE = """
Code snippet 1: $code_1
Code snippet 2: $code_2
"""

input_FT = """
Code snippet 1: $code_1
Code snippet 2: $code_2 
"""

output_FT = """
$output
"""


class PromptTemplate:
    def __init__(self):
        pass

    def get_simple_template(self, instruction_template, response_template):
        return Template(instruction_template  + Simple_template + input_PE + response_template)

    def get_input_template_FT(self, instruction_template, response_template):
        return Template(instruction_template + Simple_template + input_FT + response_template + output_FT)

    def get_input_template_similarity(self, instruction_template, response_template):
        return Template(instruction_template + similarity_template + input_FT + response_template)

def build_prompt(cfg):
    instruction_template = cfg.PROMPT.INSTRUCTION_TEMPLATE
    response_template = cfg.PROMPT.RESPONSE_TEMPLATE
    if cfg.TASK.NAME == "prompt_engineering":
        return PromptTemplate().get_simple_template(instruction_template, response_template)
    elif cfg.TASK.NAME == "fine_tuning":

        return PromptTemplate().get_input_template_FT(instruction_template, response_template)
    elif cfg.TASK.NAME == "similarity_score":
        return PromptTemplate().get_input_template_similarity(instruction_template, response_template)
    else:
        print("Unknown task")


if __name__ == '__main__':
    print(PromptTemplate().get_input_template_similarity('###Instruction:', '###Response:').template)
