"""
Dataset distillation for learning-to-rank
"""

import logging
import pandas as pd
import wandb
from fire import Fire
from munch import Munch, unmunchify
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from utils.utils import update_config, create_logging_fn, make_directories, save_results
from dltr.trainer import DistilledTrainer


def main(yaml_file, **kwargs):
    configs = Munch.fromYAML(open(yaml_file, 'rb'))

    if kwargs:
        configs = update_config(configs, kwargs)

    print(configs)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    project = "DistillLtR-SIGIR"

    if configs.General.Wandb:
        run = wandb.init(
            project=project,
            dir=configs.Datasets.Full.Location,
            name=configs.Experiment.Name,
            config=unmunchify(configs),
            tags=[configs.Datasets.Full.ID, configs.Experiment.CodeTag]
        )
        base = "./"  # If you are running your experiments in another server, set this as the project root directory.
        if configs.Datasets.Full.UseLocalFiles:
            base = "./"
        wandb.run.log_code(root=base,
                           include_fn=lambda path:
                           path.startswith(base+"configs") or path.startswith(base+"datasets") or
                           path.startswith(base+"dltr") or path.startswith(base+"loss_functions") or
                           path.startswith(base+"models") or path.startswith(base+"utils") or
                           path.endswith(base+"main.py"))
    make_directories(configs)

    configs.logger = create_logging_fn(configs.General.Wandb)

    trainer = DistilledTrainer(configs)
    trainer.run()

    if configs.General.Wandb:
        save_results(pd.DataFrame.from_records([dict(wandb.summary)]), os.path.join(configs.General.SaveDir,
                                                                                    "summary.csv"))


if __name__ == "__main__":
    Fire(main)
