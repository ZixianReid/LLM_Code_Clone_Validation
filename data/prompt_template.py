from string import Template

prefix_ft = "<s>[INST]"

Simple_template = """
Please analyze the following two code snippets and determine if they are code clones. Respond 
with only ###RESULT###@@YES@@ for clones or ###RESULT###@@NO@@ if not. Provide no other output.
"""

input_PE = """
Code snippet 1: $code_1
Code snippet 2: $code_2
[INST]
</s>
"""

input_FT = """
Code snippet 1: $code_1
Code snippet 2: $code_2
[/INST] 
$output
</s>
"""


class PromptTemplate:
    def __init__(self):
        pass

    def get_simple_template(self):
        return Template(prefix_ft + "\n" + Simple_template + "\n" + input_PE)

    def get_input_template_FT(self):
        return Template(prefix_ft + "\n" + Simple_template + "\n" + input_FT)


def build_prompt(cfg):
    if cfg.TASK.NAME == "prompt_engineering":
        return PromptTemplate().get_simple_template()
    elif cfg.TASK.NAME == "fine_tuning":
        return PromptTemplate().get_input_template_FT()
    else:
        print("Unknown task")


if __name__ == '__main__':
    print(Template(prefix_ft + "\n" + Simple_template + "\n" + input_FT).template)
