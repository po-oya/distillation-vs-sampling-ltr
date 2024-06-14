from typing import Union, Type
from dataclasses import dataclass
import torch
from torch.optim import Adam, SGD
from pytorchltr.datasets import MSLR30K, MSLR10K, IstellaS, IstellaX
from dltr.alg import GradientMatchingDistillation, DistributionMatchingDistillation
from datasets.yahoo import Yahoo
from pytorchltr.loss import LambdaNDCGLoss2
from loss_functions.loss_func import MSEGradientMatchLoss, CosGradientMatchLoss, PointWiseRegLoss


__mappings__ = {
    # Algorithms:
    "GM": GradientMatchingDistillation,  # Gradient matching distillation
    "DM": DistributionMatchingDistillation,  # Distribution matching distillation
    # Datasets:
    "MSLR10K": MSLR10K,
    "MSLR30K": MSLR30K,
    "YAHOO": Yahoo,
    "ISTELLAS": IstellaS,
    "ISTELLAX": IstellaX,
}


__losses__ = {
    "POINTWISEREGLOSS": PointWiseRegLoss,
    "LambdaNDCGLoss2": LambdaNDCGLoss2,
    "MSEGRADIENTMATCHLOSS": MSEGradientMatchLoss,
    "COSGRADIENTMATCHLOSS": CosGradientMatchLoss

}


@dataclass
class TrainerLoss:
    distillation = None
    ranking = None


def mean_ranking_loss(ranking_loss_fn):

    def ranking_loss(*args, **kwargs):
        return ranking_loss_fn(*args, **kwargs).mean()

    return ranking_loss


def prepare_losses(configs) -> TrainerLoss:
    matching_loss = __losses__[configs.Trainer.DistillLoss]()
    ranking_loss = __losses__[configs.Trainer.RankingLoss]()

    if configs.Trainer.RankingLoss != "POINTWISEREGLOSS":
        ranking_loss = mean_ranking_loss(ranking_loss)

    trainer_loss = TrainerLoss()
    trainer_loss.distillation = matching_loss
    trainer_loss.ranking = ranking_loss

    return trainer_loss


def mappings(key_tag: str):
    if key_tag not in __mappings__.keys():
        raise KeyError("{} does not exist in the existing mappings".format(key_tag))
    return __mappings__[key_tag]


__mapping__feature_dimensions = {
    "MSLR10K": 136,
    "MSLR30K": 136,
    "ISTELLAS": 220,
    "ISTELLAX": 220,
    "YAHOO": 699
    }


def get_feature_dim(dataset_name: object) -> object:
    return __mapping__feature_dimensions[dataset_name]


def get_optimizer(name: str) -> Type[Union[Adam, SGD]]:
    __optimizers__ = {
        "ADAM": torch.optim.Adam,
        "SGD": torch.optim.SGD
    }
    return __optimizers__[name]
