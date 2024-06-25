from datasets import Dataset
from datasets import load_dataset
from string import Template


class MainDataset:

    def __init__(self, name, prompt, cache_dir):
        super().__init__()
        self.dataset = load_dataset(name, cache_dir=cache_dir)
        self.dataset_train = self.dataset['train']
        self.dataset_test = self.dataset['test']
        self.dataset_val = self.dataset['validation']
        self.instruction_template = prompt

    def __contact_input_prompt(self, dataset):
        dataset = dataset.map(
            lambda example: {'prompt_input': self.instruction_template.substitute({"code_1": example['func1'],
                                                                                   "code_2": example['func2']})}
        )
        return dataset

    def __contact_output_fine_tuning(self, dataset):
        output = Template("(output) $output")
        fine_tuning_template_string = self.instruction_template.template + output.template  # Add the string values
        fine_tuning_template = Template(fine_tuning_template_string)  # Convert back to

        dataset = dataset.map(
            lambda example: {
                'prompt_input': fine_tuning_template.substitute({
                    "code_1": example['func1'],
                    "code_2": example['func2'],
                    "output": 'Yes' if example['label'] == 1 else 'No'
                })
            }
        )
        return dataset

    def build(self):
        # map to specific column
        self.dataset_train = self.__contact_output_fine_tuning(self.dataset_train)
        self.dataset_test = self.__contact_output_fine_tuning(self.dataset_test)
        self.dataset_val = self.__contact_output_fine_tuning(self.dataset_val)

    def build_prompt(self):
        self.dataset_test = self.__contact_input_prompt(self.dataset_test)


class BCBDataset(MainDataset):
    def __init__(self, name, prompt, cache_dir):
        super().__init__(name, prompt, cache_dir)
        self.dataset_test = self.dataset['test']


class OJCloneDataset(MainDataset):
    def __init__(self, name, prompt, cache_dir):
        super().__init__(name, prompt, cache_dir)
        self.dataset_test = self.dataset['test']


__REGISTERED_DATASETS = {"Reid996/big_clone_bench": BCBDataset,
                         "Reid996/OJClone_code_clone_unbalanced": OJCloneDataset}


def build_dataset(cfg, prompt):
    cache_dir = cfg.TASK.CACHE_DIR
    dataset = __REGISTERED_DATASETS[cfg.DATA.NAME](cfg.DATA.NAME, prompt, cache_dir)
    if cfg.TASK.NAME == "prompt_engineering":
        dataset.build_prompt()
    elif cfg.TASK.NAME == "fine_tuning":
        dataset.build()
    else:
        print("Unknown task")

    return dataset
