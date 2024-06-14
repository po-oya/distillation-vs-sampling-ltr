import numpy as np
import random
import torch
from utils.ds import mappings, get_feature_dim
from dltr.alg import LtRDistillation
from datasets.interface import FullDataset, DistilledDataset, DatasetInterfaces
from models.mlp import MLPLtR


class BaseTrainer:
    def __init__(self, configs):
        self.datasets = DatasetInterfaces
        self.configs = configs
        # Config the device
        print("CONFIG DEVICE: ", configs.General.Device)
        if self.configs.General.Device == "cuda" and torch.cuda.is_available():
            self.configs.General.Device = "cuda:0"
            print("CUDA IS AVAILABLE ...")
        else:
            print("CUDA IS -NOT- AVAILABLE ...")
            self.configs.General.Device = torch.device("cpu")

        # Config random states
        self.__init_random_states()

        # Initialize LtR model
        self.ltr_model = self.__init_ltr_model()

        self.configs.datasets = self.datasets
        self.configs.ltr_net = self.ltr_model

        self.alg = None

    def __init_random_states(self):
        if self.configs.General.Deterministic:
            np.random.seed(self.configs.General.Seed)
            random.seed(self.configs.General.Seed)
            torch.manual_seed(self.configs.General.Seed)
            torch.backends.cudnn.deterministic = True

    def __init_ltr_model(self) -> torch.nn.Module:
        if self.configs.Trainer.ModelType == "MLP":
            scale = self.configs.Trainer.ModelScale
            self.configs.Datasets.Full.FeatureSize = get_feature_dim(self.configs.Datasets.Full.ID)
            if scale == 1:
                dimensions = [self.configs.Datasets.Full.FeatureSize, 100, 50]
            elif scale == 2:
                dimensions = [self.configs.Datasets.Full.FeatureSize, 250, 150, 75]
            elif scale == 3:
                dimensions = [self.configs.Datasets.Full.FeatureSize, 400, 200, 100, 50]
            else:
                raise NotImplementedError

            self.configs.General.ModelDim = dimensions
            model = MLPLtR(dimensions)
            model.to(self.configs.General.Device)
            return model
        else:
            raise NotImplementedError

    def init_original_dataset(self):
        location = self.configs.Datasets.Full.Location
        dataset_kwargs = {
            "normalize": False,
            "filter_queries": self.configs.Datasets.Full.FilterQueries,
            "location": location
        }
        if self.configs.Datasets.Full.ID in ["YAHOO"]:
            dataset_kwargs["location"] = location

        if self.configs.Datasets.Full.Normalization:
            if self.configs.Datasets.Full.NormalizationType == "Query":
                dataset_kwargs["normalize"] = True
            else:
                dataset_kwargs["normalize"] = False

        if self.configs.Datasets.Full.UseLocalFiles is False:
            dataset_kwargs["location"] = location + self.configs.Datasets.Full.ID + "/"

        if self.configs.Datasets.Full.ID in ["MSLR10K", "MSLR30K"]:
            dataset_kwargs["fold"] = self.configs.Datasets.Full.fold
        if self.configs.Datasets.Full.ID not in ["YAHOO"]:
            dataset_kwargs["validate_checksums"] = False
            dataset_kwargs["download"] = False

        train_data = mappings(self.configs.Datasets.Full.ID)(split="train", **dataset_kwargs)
        valid_data = mappings(self.configs.Datasets.Full.ID)(split="vali", **dataset_kwargs)
        test_data = mappings(self.configs.Datasets.Full.ID)(split="test", **dataset_kwargs)


        self.datasets.ValidFull = FullDataset(self.configs, **{"data": valid_data, "split": "valid"})
        self.datasets.TestFull = FullDataset(self.configs, **{"data": test_data, "split": "test"})

        full_kwargs = {
            "data": train_data,
            "Normalization": self.configs.Datasets.Full.NormalizationType,
            "split": "train",
        }
        self.datasets.TrainFull = FullDataset(self.configs, **full_kwargs)

    def pre_training_analysis(self):
        self.alg.pre_training_analysis()

    def train(self):
        self.alg.train()

    def evaluate(self):
        self.alg.evaluate()

    def run(self):
        self.pre_training_analysis()
        self.train()
        self.evaluate()


class DistilledTrainer(BaseTrainer):

    def __init__(self, configs):
        super(DistilledTrainer, self).__init__(configs)

        # Initialize original dataset (train, valid, test)
        self.init_original_dataset()
        # Initialize distilled dataset
        self.__init_distilled_dataset()

        # Initialize distillation algorithm
        self.alg: LtRDistillation = mappings(self.configs.General.Alg)(self.configs)

    def __init_distilled_dataset(self):
        distill_kwargs = {
            "full_data": self.datasets.TrainFull,
        }
        self.datasets.Distilled = DistilledDataset(self.configs, **distill_kwargs)
