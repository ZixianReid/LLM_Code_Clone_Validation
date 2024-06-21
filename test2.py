import argparse
# from data.datasets_group import BCBDataset
from config import cfg
from data import build_dataset, build_prompt
from engine import build_prompt_engineering
from utils.utils import print_info

def run(cfg):
    # build prompt
    prompt = build_prompt(cfg)
    # build dataset
    dataset = build_dataset(cfg, prompt)

    # build model
    if cfg.TASK.NAME == 'prompt_engineering':
        model = build_prompt_engineering(cfg)
        model.run(cfg, dataset)
    elif cfg.TASK.NAME == 'fine_tuning':
        pass
    else:
        print('Unknown task')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default='/home/zixian/PycharmProjects/LLM_Code_Clone_Validation/config/codellama_7b_one_shot_ojclone.yaml')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print_info(cfg)

    run(cfg)


if __name__ == '__main__':
    main()
