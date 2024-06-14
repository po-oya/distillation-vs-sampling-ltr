import os
from operator import attrgetter
import datetime
import json
import uuid
import logging
import pandas as pd
import wandb


def exp_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_json(js_dict, path):
    """

    :param js_dict: Dictionary to be saved
    :param path: full json path
    :return:
    """
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(js_dict, file, ensure_ascii=False, indent=4)


def make_directories(config):
    base = config.General.SaveDir
    if base == "./":
        base = os.getcwd()

    base = exp_mkdir(base)
    dataset_id: str = config.Datasets.Full.ID.lower()
    exp_name: str = config.Experiment.Name.lower()
    base_this_exp = exp_mkdir(os.path.join(base, exp_name))

    ct = datetime.datetime.now()
    ct_date = ct.date()
    ct_time = ct.time()
    exp_dir_rand_name = str(ct_date) + "$" + str(ct_time) + "$" + str(uuid.uuid4())

    config.General.SaveDir = exp_mkdir(os.path.join(base_this_exp, exp_dir_rand_name))

    # Save configurations
    save_json(config, os.path.join(config.General.SaveDir, "config.json"))


def save_results(df: pd.DataFrame, path):
    df.to_csv(path)


def update_config(config, kwargs):
    """
    Update config with flags from the commandline
    :param config:
    :param kwargs:
    :return:
    """

    for key, value in kwargs.items():
        try:
            _ = attrgetter(key)(config)
            subconfig = config

            for sub_key in key.split('.'):
                if isinstance(subconfig[sub_key], dict):
                    subconfig = subconfig[sub_key]
                else:
                    if isinstance(subconfig[sub_key], type(value)):
                        subconfig[sub_key] = value
                    else:
                        raise Exception("wrong value type")

        except AttributeError:
            print("{} not in config".format(key))

    return config


def create_logging_fn(log_for_wandb: bool):

    def logging_fn(msg, wandb_log: bool, wandb_loggers: dict = None, step=None):
        if msg:
            logging.info(msg)
        if log_for_wandb and wandb_log:
            if step:
                wandb.log(wandb_loggers, step)
            else:
                wandb.log(wandb_loggers)

    return logging_fn
