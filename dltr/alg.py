import torch
import os
import pandas as pd
import numpy as np
from math import ceil
from time import time
from copy import deepcopy
from torch.utils.data import DataLoader
from pytorchltr.evaluation import ndcg, arp
from utils import ds as uds
from utils.utils import save_results, exp_mkdir
from models.mlp import MLPLtR
from datasets.interface import ltr_collate_fn, DatasetInterfaces, DistilledDataset


class LtRExperiment:

    def __init__(self, configs):
        self.configs = configs
        self.ltr_net = self.configs.ltr_net
        self.datasets: DatasetInterfaces = self.configs.datasets
        self.loss = uds.prepare_losses(self.configs)
        self.logging_fn = configs.logger
        self.metric_results = {}
        self.valid_loader = DataLoader(self.datasets.ValidFull, batch_size=self.configs.Datasets.Full.Bsz,
                                       shuffle=False, collate_fn=ltr_collate_fn(self.configs.General.Device))
        base_valid_status = {"best_metric": None, "best_step": None}
        self.valid_status = {"overall": base_valid_status.copy(), "per_instance": base_valid_status.copy()}

    def train_ltr(self, ltr_net: MLPLtR, loader: DataLoader):
        self.logging_fn("Training a LTR model ...", False)
        optimizer = torch.optim.Adagrad(ltr_net.parameters(), lr=self.configs.Trainer.RankingLR)
        crt = self.loss.ranking

        loss_train_ltr_net = []

        best_ltr_net = None
        best_itr_loss = []
        best_ndcg10_val = 0
        best_itr = 0

        for i in range(1, self.configs.Trainer.Epochs + 1):
            ltr_net.train()
            batch_loss = 0
            for batch in loader:
                xs, ys, n = batch[0], batch[1], batch[2]
                pred = ltr_net(xs)
                loss = crt(pred, ys, n)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            loss_train_ltr_net.append(batch_loss / len(loader))

            if i % 2 == 0:
                metrics_queries_validation = self.eval_ltr(ltr_net, self.valid_loader,
                                                           length=len(self.datasets.ValidFull))
                ndcg_10_val = torch.mean(metrics_queries_validation["ndcg10"])
                if (best_ltr_net is None) or (ndcg_10_val > best_ndcg10_val):
                    best_ltr_net = ltr_net
                    best_ndcg10_val = ndcg_10_val
                    best_itr_loss = loss_train_ltr_net
                    best_itr = i
                elif i - best_itr >= 4:
                    break

        return best_ltr_net, best_itr_loss

    @staticmethod
    def eval_ltr(ltr_net: torch.nn.Module, loader: DataLoader, length):
        ltr_net.eval()
        ndcg_3_queries = []
        ndcg_10_queries = []
        arp_queries = []

        for batch in loader:
            xs, ys, n = batch[0], batch[1], batch[2]
            pred = ltr_net(xs)
            ndcg_10_queries.append(ndcg(pred, ys, n, k=10))
            ndcg_3_queries.append(ndcg(pred, ys, n, k=3))
            arp_queries.append(arp(pred, ys, n))
        ndcg_10_queries = torch.cat(ndcg_10_queries, dim=0)
        ndcg_3_queries = torch.cat(ndcg_3_queries, dim=0)
        arp_queries = torch.cat(arp_queries, dim=0)

        return {"ndcg10": ndcg_10_queries, "arp": arp_queries, "ndcg3": ndcg_3_queries}

    def log_metrics_per_queries(self):
        self.logging_fn("Logging metrics ...", False)
        metrics_df = pd.DataFrame.from_dict(self.metric_results, orient="index")
        metrics_df = metrics_df.transpose()
        save_results(metrics_df, os.path.join(self.configs.General.SaveDir, "metrics.csv"))
        qd_vecs_ds = self.datasets.Distilled.total_vecs

        wandb_loggers = {
            "total_qd_vecs_full": self.datasets.TrainFull.total_vecs,
            "total_qd_vecs_ds": qd_vecs_ds
        }
        msg = "Total qd vecs in full data: {} and total qd vecs in distilled/sampled data: {}".format(
            self.datasets.TrainFull.total_vecs, qd_vecs_ds
        )
        self.logging_fn(msg, True, wandb_loggers)

    def pre_training_analysis(self):
        raise NotImplementedError

    def validate(self, itr, stage):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class LtRDistillation(LtRExperiment):

    def __init__(self, configs):
        super(LtRDistillation, self).__init__(configs)
        self.mean_query_statistics = {"distill_init": {}, "distill_final": {}, "original": {}}
        self.validated_distilled_dataset = {"overall": None, "per_instance": None}

    def pre_training_analysis(self):
        self.logging_fn("Starting pre_training_analysis ...", False)
        self.datasets.TrainFull.get_items_query_wise = True
        self.datasets.TestFull.get_items_query_wise = True
        self.datasets.Distilled.get_items_query_wise = True
        self.distillation_evaluation(evaluation_step="InitEval")

    def distillation_evaluation(self, evaluation_step: str):
        if self.configs.Datasets.Distilled.SampleQueryWise:
            distill_bsz = len(self.datasets.Distilled)
        else:
            distill_bsz = self.configs.Datasets.Distilled.Bsz

        full_loader = DataLoader(self.datasets.TrainFull, batch_size=self.configs.Datasets.Full.Bsz, shuffle=True,
                                 collate_fn=ltr_collate_fn(self.configs.General.Device))
        test_loader = DataLoader(self.datasets.TestFull, batch_size=self.configs.Datasets.Full.Bsz, shuffle=False,
                                 collate_fn=ltr_collate_fn(self.configs.General.Device))
        distill_loader = DataLoader(self.datasets.Distilled, batch_size=distill_bsz,
                                    shuffle=False,
                                    collate_fn=ltr_collate_fn(self.configs.General.Device))
        ltr_net: MLPLtR = self.configs.ltr_net

        if evaluation_step == "InitEval":

            ltr_net, _ = self.train_ltr(ltr_net, full_loader)

            eval_full_info_with_full_test = self.eval_ltr(ltr_net, test_loader, length=len(self.datasets.TestFull))

            ndcg_10_full_info_with_full_test = eval_full_info_with_full_test["ndcg10"]
            ndcg_3_full_info_with_full_test = eval_full_info_with_full_test["ndcg3"]
            arp_full_info_with_full_test = eval_full_info_with_full_test["arp"]

            msg = "ndcg@10***ARP-{}-Mean-FullInfo - Test - {}***{}".format(
                evaluation_step, torch.mean(ndcg_10_full_info_with_full_test), torch.mean(arp_full_info_with_full_test)
            )
            self.metric_results["NDCG@10-{}-FullInfo-Test".format(evaluation_step)] = ndcg_10_full_info_with_full_test.cpu().numpy()
            self.metric_results["NDCG@3-{}-FullInfo-Test".format(evaluation_step)] = ndcg_3_full_info_with_full_test.cpu().numpy()
            self.metric_results["ARP-{}-FullInfo-Test".format(evaluation_step)] = arp_full_info_with_full_test.cpu().numpy()

            wandb_loggers = {
                "NDCG@10-{}-FullInfoTest".format(evaluation_step): torch.mean(ndcg_10_full_info_with_full_test),
                "ARP-{}-FullInfoTest".format(evaluation_step): torch.mean(arp_full_info_with_full_test)
            }
            self.logging_fn(msg, True, wandb_loggers)

        else:
            pass

        ndcg_10_runs = []
        arp_runs = []
        best_valid_ndcg_10 = 0
        best_ltr_valid = None
        for _ in range(self.configs.General.Runs):
            ltr_net.init_weights()
            ltr_net, _ = self.train_ltr(ltr_net, distill_loader)
            eval_distilled_info_with_full_valid = self.eval_ltr(ltr_net, self.valid_loader, length=len(self.datasets.ValidFull))

            ndcg_10_valid = eval_distilled_info_with_full_valid["ndcg10"].mean().item()
            arp_valid = eval_distilled_info_with_full_valid["arp"].mean().item()
            ndcg_10_runs.append(ndcg_10_valid)
            arp_runs.append(arp_valid)
            if (ndcg_10_valid > best_valid_ndcg_10) or (best_ltr_valid is None):
                best_valid_ndcg_10 = ndcg_10_valid
                best_ltr_valid = deepcopy(ltr_net)

        msg = "ndcg@10-{}-DistilledInfo- ValidationRuns - Mean/STD {}/{}".format(
            evaluation_step, np.mean(ndcg_10_runs), np.std(ndcg_10_runs)) + \
              "\nARP-{}-DistilledInfo- ValidationRuns - Mean/STD {}/{}".format(
                  evaluation_step,
                  np.mean(arp_runs), np.std(arp_runs)
              )

        wandb_loggers = {
            "NDCG@10-{}-DistilledInfoValidationBest".format(evaluation_step): best_valid_ndcg_10,
            "NDCG@10-{}-DistilledInfoValidationRuns-Mean".format(evaluation_step): np.mean(ndcg_10_runs),
            "NDCG@10-{}-DistilledInfoValidationRuns-STD".format(evaluation_step): np.std(ndcg_10_runs),
            "ARP-{}-DistilledInfoValidationRuns-Mean".format(evaluation_step): np.mean(arp_runs),
            "ARP-{}-DistilledInfoValidationRuns-STD".format(evaluation_step): np.std(arp_runs)
        }
        self.logging_fn(msg, True, wandb_loggers)

        best_ltr_valid.to(self.configs.General.Device)
        eval_distill_info_with_full_test = self.eval_ltr(best_ltr_valid, test_loader, len(self.datasets.TestFull))
        ndcg_10_distill_info_with_full_test = eval_distill_info_with_full_test["ndcg10"]
        ndcg_3_distill_info_with_full_test = eval_distill_info_with_full_test["ndcg3"]
        arp_distill_info_with_full_test = eval_distill_info_with_full_test["arp"]
        self.metric_results["NDCG@10-{}-DistillInfoTest".format(
            evaluation_step)] = ndcg_10_distill_info_with_full_test.cpu().numpy()
        self.metric_results["NDCG@3-{}-DistillInfoTest".format(
            evaluation_step)] = ndcg_3_distill_info_with_full_test.cpu().numpy()
        self.metric_results[
            "ARP-{}-DistillInfoTest".format(evaluation_step)] = arp_distill_info_with_full_test.cpu().numpy()
        msg = "ndcg@10-{}-DistillInfo- Test - Mean Values {}".format(
            evaluation_step, ndcg_10_distill_info_with_full_test.mean()) + \
            "\nARP-{}-DistillInfo- Distill/Train/Test - Mean Values {}".format(
                  evaluation_step, arp_distill_info_with_full_test.mean())

        wandb_loggers = {
            "NDCG@10-{}-DistillInfoTest-Mean".format(evaluation_step): ndcg_10_distill_info_with_full_test.mean(),
            "ARP-{}-DistillInfoTest-Mean".format(evaluation_step): arp_distill_info_with_full_test.mean()
        }
        self.logging_fn(msg, True, wandb_loggers)

        if evaluation_step == "InitEval":
            df_dict = {
                "qid": [],
                "dataset": [],
                "sampling_ratio": [],
                "distillation_alg": [],
                "R0": [],
                "R1": [],
                "R2": [],
                "R3": [],
                "R4": [],
            }
            for item in self.datasets.Distilled:
                df_dict["qid"].append(item[3])
                df_dict["dataset"].append(self.configs.Datasets.Full.ID)
                df_dict["sampling_ratio"].append(self.configs.Datasets.Distilled.CompressionRatio)
                df_dict["distillation_alg"].append(self.configs.Experiment.Name)
                unique, counts = np.unique(item[1], return_counts=True)
                stats = dict(zip(unique, counts))
                df_dict["R0"].append(stats.get(0, 0))
                df_dict["R1"].append(stats.get(1, 0))
                df_dict["R2"].append(stats.get(2, 0))
                df_dict["R3"].append(stats.get(3, 0))
                df_dict["R4"].append(stats.get(4, 0))

            df_qrels = pd.DataFrame.from_dict(df_dict)
            self.logging_fn("Logging qrels ...", False)
            save_results(df_qrels, os.path.join(self.configs.General.SaveDir, "df_qrels.csv"))
            print(self.configs.General.SaveDir)

    def validate(self, itr, stage):
        if stage == "overall":
            valid_freq = self.configs.General.ValidFreq
            valid_patience = self.configs.General.ValidPatience
        elif stage == "per_instance":
            valid_freq = self.configs.General.ValidFreqInstance
            valid_patience = self.configs.General.ValidPatienceInstance
        else:
            raise NotImplementedError

        if itr % valid_freq == 0:
            valid_ltr_net = MLPLtR(self.configs.General.ModelDim)
            valid_ltr_net.load_state_dict(self.configs.ltr_net.state_dict())
            valid_ltr_net.to(self.configs.General.Device)
            distill_loader = DataLoader(self.datasets.Distilled, batch_size=len(self.datasets.Distilled),
                                        shuffle=False,
                                        collate_fn=ltr_collate_fn(self.configs.General.Device))
            valid_ltr_net, _ = self.train_ltr(valid_ltr_net, distill_loader)
            metrics_queries = self.eval_ltr(valid_ltr_net, self.valid_loader, length=len(self.datasets.ValidFull))
            ndcg_10_val = torch.mean(metrics_queries["ndcg10"])
            if (self.valid_status[stage]["best_metric"] is None) or (ndcg_10_val >
                                                                     self.valid_status[stage]["best_metric"]):
                self.validated_distilled_dataset[stage] = deepcopy(self.datasets.Distilled)
                self.valid_status[stage]["best_metric"], self.valid_status[stage]["best_step"] = ndcg_10_val, itr

            elif itr - self.valid_status[stage]["best_step"] >= valid_patience:
                return True

        return False

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        self.logging_fn("Starting post_training_analysis ...", False)
        self.datasets.TrainFull.get_items_query_wise = True
        self.datasets.TestFull.get_items_query_wise = True
        self.datasets.Distilled.get_items_query_wise = True
        self.distillation_evaluation(evaluation_step="FinalEval")
        self.log_metrics_per_queries()


class GradientMatchingDistillation(LtRDistillation):

    def __init__(self, configs):
        super(GradientMatchingDistillation, self).__init__(configs)

    def train(self):
        torch.autograd.set_detect_anomaly(True)

        if self.configs.Trainer.LabelWiseDistillation:
            # In the label_wise mode the following settings are required.
            assert self.configs.Trainer.MultiBatchDistilledData is True
            assert self.configs.Datasets.Distilled.SampleQueryWise is False
            self.datasets.Distilled.get_items_query_wise = False
            self.datasets.TrainFull.get_items_query_wise = False

        distill_batch = self.datasets.Distilled.collate_full_batch()
        x_distill, y_distill, n_distill, qid_distill = distill_batch[0], distill_batch[1], distill_batch[2] \
            , distill_batch[3]
        # Transferring to the right device is assumed to be done in collate_fn

        distill_optimizer: torch.optim.Optimizer = uds.get_optimizer(self.configs.Trainer.Optimizer)(
            [x_distill], lr=self.configs.Trainer.DistillLR)

        t_distill_begin = time()

        if self.configs.Trainer.MultiBatchDistilledData:
            original_data_total_batch_iterations = ceil(len(self.datasets.TrainFull.query_wise_reading_items) /
                                                        self.configs.Trainer.BatchSize)
            distill_batch_size = int(x_distill.shape[0]/original_data_total_batch_iterations)
        else:
            distill_batch_size = x_distill.shape[0]
            original_data_total_batch_iterations = None

        log_loss = {"distill_loss_over_batches": [], "distill_loss_over_epochs": [], "distill_loss_over_models": []}
        for model_init_itr in range(self.configs.Trainer.ModelInstances):
            self.ltr_net.init_weights()
            self.ltr_net.train()
            full_loader = torch.utils.data.DataLoader(self.datasets.TrainFull,
                                                      batch_size=self.configs.Trainer.BatchSize,
                                                      shuffle=self.configs.Trainer.Suffle,
                                                      collate_fn=ltr_collate_fn(self.configs.General.Device))

            model_init_distill_loss_list = []
            for epoch_itr in range(self.configs.Trainer.GMEpochs):
                x_distill.requires_grad = True
                distill_loss = torch.tensor(0.0).to(self.configs.General.Device)

                distill_batch_itr = 0

                for batch_itr, batch_full in enumerate(full_loader):
                    self.ltr_net.zero_grad()
                    x_full, y_full, n_full = batch_full[0], batch_full[1], batch_full[2]

                    out_full = self.ltr_net(x_full)
                    loss_full = self.loss.ranking(out_full, y_full, n_full).mean()
                    grad_full = torch.autograd.grad(loss_full, list(self.ltr_net.parameters()), create_graph=True)
                    grad_full = list((_.clone() for _ in grad_full))

                    distill_batch_start_itr = distill_batch_itr
                    distill_batch_end_itr = distill_batch_itr + distill_batch_size
                    if self.configs.Trainer.MultiBatchDistilledData:
                        if original_data_total_batch_iterations == batch_itr + 1:
                            distill_batch_end_itr = len(x_distill)  # making sure the last queries
                            # are placed in final batch
                    x_distill_batch = x_distill[distill_batch_start_itr:distill_batch_end_itr]
                    y_distill_batch = y_distill[distill_batch_start_itr:distill_batch_end_itr]
                    n_distill_batch = n_distill[distill_batch_start_itr:distill_batch_end_itr]
                    out_distill = self.ltr_net(x_distill_batch)
                    loss_distill = self.loss.ranking(out_distill, y_distill_batch, n_distill_batch).mean()
                    grad_distill = torch.autograd.grad(loss_distill, list(self.ltr_net.parameters()), create_graph=True)
                    grad_distill = list((_.clone() for _ in grad_distill))

                    batch_loss = self.loss.distillation(grad_distill, grad_full)
                    # distill_loss += batch_loss
                    distill_loss += batch_loss.item()
                    #
                    distill_optimizer.zero_grad()
                    batch_loss.backward(inputs=x_distill)
                    distill_optimizer.step()

                    if self.configs.Trainer.MultiBatchDistilledData:
                        distill_batch_itr += distill_batch_size

                    msg = "{}/{} batches in epoch {} and model instance {}".format(batch_itr, len(full_loader),
                                                                                   epoch_itr, model_init_itr)
                    wandb_loggers = {
                        "distill_loss_over_batches": batch_loss.item()
                    }
                    self.logging_fn(msg, True, wandb_loggers, step=None)
                    log_loss["distill_loss_over_batches"].append(batch_loss.item())

                print("Epoch optimization!")

                # do training of ltr_net based on distill_data
                x_distill.requires_grad = False
                if self.configs.Trainer.LabelWiseDistillation:
                    self.datasets.Distilled.update_label_wise([x_distill, y_distill, n_distill, qid_distill])
                    self.datasets.Distilled.get_items_query_wise = True  # Temporarily set query_wise_mode on for
                    # training ltr_net
                else:
                    self.datasets.Distilled.update_query_wise([x_distill, y_distill, n_distill, qid_distill])

                distill_loader = DataLoader(self.datasets.Distilled, batch_size=len(self.datasets.Distilled),
                                            shuffle=False,
                                            collate_fn=ltr_collate_fn(self.configs.General.Device))
                self.ltr_net, _ = self.train_ltr(self.ltr_net, distill_loader)
                if self.configs.Trainer.LabelWiseDistillation:
                    self.datasets.Distilled.get_items_query_wise = False  # Set label_wise_mode on for the reset of
                    # distillation

                msg = "Training GradMatch - {}/{} ModelInits {}/{} Epochs - Epoch Distill Loss {}".format(
                    model_init_itr + 1, self.configs.Trainer.ModelInstances, epoch_itr + 1,
                    self.configs.Trainer.GMEpochs, distill_loss.item() / len(full_loader))
                wandb_loggers = {
                    "distill_loss_over_epochs": distill_loss.item() / len(full_loader)
                }
                self.logging_fn(msg, True, wandb_loggers, step=None)
                log_loss["distill_loss_over_epochs"].append(distill_loss.item() / len(full_loader))

                model_init_distill_loss_list.append(distill_loss.item() / len(full_loader))

                # Decide for early stop:
                early_stop_per_instance = self.validate(epoch_itr, "per_instance")
                if early_stop_per_instance or (epoch_itr == self.configs.Trainer.GMEpochs - 1):
                    self.datasets.Distilled = self.validated_distilled_dataset["per_instance"]
                    wandb_loggers = {
                        "best_step_per_instance": self.valid_status["per_instance"]["best_step"],
                        "best_valid_ndcg10_per_instance": self.valid_status["per_instance"]["best_metric"]
                    }
                    self.logging_fn("Exiting early in model instance {} - best_setp_per_instance: "
                                    "{} - best_valid_ndcg10per_instance: {} ..."
                                    .format(model_init_itr, self.valid_status["per_instance"]["best_step"],
                                            self.valid_status["per_instance"]["best_metric"]),
                                    True, wandb_loggers)
                    self.valid_status["per_instance"]["best_metric"] = None
                    self.valid_status["per_instance"]["best_step"] = None
                    break  # If validate returns True -> early stopping
                # ############

            wandb_loggers = {
                "distill_loss_over_models": sum(model_init_distill_loss_list) / len(model_init_distill_loss_list)
            }
            self.logging_fn(None, True, wandb_loggers, step=None)
            log_loss["distill_loss_over_models"].append(
                sum(model_init_distill_loss_list) / len(model_init_distill_loss_list))

            # Validation here:
            early_stop = self.validate(model_init_itr, "overall")
            if early_stop or (model_init_itr == self.configs.Trainer.ModelInstances - 1):
                self.datasets.Distilled = self.validated_distilled_dataset["overall"]
                wandb_loggers = {
                    "best_step": self.valid_status["overall"]["best_step"],
                    "best_valid_ndcg10": self.valid_status["overall"]["best_metric"]
                }
                self.logging_fn("Exiting early - best_setp: {} - best_valid_ndcg10: {} ..."
                                .format(self.valid_status["overall"]["best_step"],
                                        self.valid_status["overall"]["best_metric"]),
                                True, wandb_loggers)
                break  # If validate returns True -> early stopping
            # ############

        # After distillation is finished both datasets should be in get_items_query_wise mode, NOTE that
        # label_wise_reading_items attribute in Distilled data is any point during distillation because it will not be
        # used.
        self.datasets.Distilled.get_items_query_wise = True
        self.datasets.TrainFull.get_items_query_wise = True

        t_distill_end = time()
        t_distill = t_distill_end - t_distill_begin
        msg = "Performing distillation took: {}".format(t_distill)
        wandb_loggers = {
            "t_distill": t_distill
        }
        self.logging_fn(msg, True, wandb_loggers)

        loss_df = pd.DataFrame.from_dict(log_loss, orient="index")
        loss_df = loss_df.transpose()
        save_results(loss_df, os.path.join(self.configs.General.SaveDir, "log_loss.csv"))


class DistributionMatchingDistillation(LtRDistillation):

    def __init__(self, configs):
        self.configs = configs
        super(DistributionMatchingDistillation, self).__init__(configs)
        pass

    def train(self):
        torch.autograd.set_detect_anomaly(True)

        if self.configs.Trainer.LabelWiseDistillation:
            # In the label_wise mode the following settings are required.
            assert self.configs.Trainer.MultiBatchDistilledData is True
            assert self.configs.Datasets.Distilled.SampleQueryWise is False
            self.datasets.Distilled.get_items_query_wise = False
            self.datasets.TrainFull.get_items_query_wise = False
        else:
            raise NotImplementedError("Distribution matching is only implemented in label_wise mode currently!")

        distill_batch = self.datasets.Distilled.collate_full_batch()
        x_distill, y_distill, n_distill, qid_distill = distill_batch[0], distill_batch[1], distill_batch[2] \
            , distill_batch[3]
        # Transferring to the right device is assumed to be done in collate_fn

        distill_optimizer: torch.optim.Optimizer = uds.get_optimizer(self.configs.Trainer.Optimizer)(
            [x_distill], lr=self.configs.Trainer.DistillLR)

        t_distill_begin = time()

        if self.configs.Trainer.MultiBatchDistilledData:
            distill_batch_size = self.configs.Trainer.BatchSize
        else:
            distill_batch_size = x_distill.shape[0]

        models_distill_loss_list = []

        log_loss = {"distill_loss_over_batches": [], "distill_loss_over_models": []}
        for model_init_itr in range(self.configs.Trainer.ModelInstances):
            self.ltr_net.init_weights()
            self.ltr_net.train()
            for param in list(self.ltr_net.parameters()):
                param.requires_grad = False

            distill_loss = torch.tensor(0.0).to(self.configs.General.Device)

            full_loader = torch.utils.data.DataLoader(self.datasets.TrainFull,
                                                      batch_size=self.configs.Trainer.BatchSize,
                                                      shuffle=self.configs.Trainer.Suffle,
                                                      collate_fn=ltr_collate_fn(self.configs.General.Device))

            x_distill.requires_grad = True
            distill_batch_itr = 0
            for batch_itr, batch_full in enumerate(full_loader):
                # self.ltr_net.zero_grad()
                x_full, y_full, n_full = batch_full[0], batch_full[1], batch_full[2]
                x_distill_batch = x_distill[distill_batch_itr:distill_batch_itr + distill_batch_size]
                n_distill_batch = n_distill[distill_batch_itr:distill_batch_itr + distill_batch_size]

                out_full = self.ltr_net(x_full, return_one_to_the_last_layer=True)
                out_distill = self.ltr_net(x_distill_batch, return_one_to_the_last_layer=True)

                mask_full = torch.arange(1, out_full.shape[1] + 1, device=self.configs.General.Device).repeat(out_full.shape[0], 1).unsqueeze(2)
                mask_distill = torch.arange(1, out_distill.shape[1]+1, device=self.configs.General.Device).repeat(out_distill.shape[0], 1).unsqueeze(2)

                n_full = n_full.unsqueeze(1).repeat(1, out_full.shape[1]).unsqueeze(2)
                n_distill_batch = n_distill_batch.unsqueeze(1).repeat(1, out_distill.shape[1]).unsqueeze(2)

                mask_full[mask_full <= n_full] = 1
                mask_distill[mask_distill <= n_distill_batch] = 1

                mask_full[mask_full > n_full] = 0
                mask_distill[mask_distill > n_distill_batch] = 0

                batch_loss = torch.sum((torch.sum(out_full * mask_full, dim=1) / torch.sum(mask_full) -
                                        torch.sum(out_distill * mask_distill, dim=1) / torch.sum(mask_distill)) ** 2)

                distill_loss += batch_loss

                if self.configs.Trainer.MultiBatchDistilledData:
                    distill_batch_itr += distill_batch_size

                wandb_loggers = {
                    "distill_loss_over_batches": batch_loss.item()
                }
                self.logging_fn(None, True, wandb_loggers)
                log_loss["distill_loss_over_batches"].append(batch_loss.item())

            ##################
            # Optimization
            distill_optimizer.zero_grad()
            distill_loss.backward()
            distill_optimizer.step()

            msg = "Training DistributionMatch - {}/{} ModelInits - Distill Loss {}".format(
                model_init_itr + 1, self.configs.Trainer.ModelInstances,
                distill_loss.item() / len(full_loader)
            )
            wandb_loggers = {
                "distill_loss_over_models": distill_loss.item() / len(full_loader)
            }
            self.logging_fn(msg, True, wandb_loggers)
            log_loss["distill_loss_over_models"].append(distill_loss.item() / len(full_loader))

            models_distill_loss_list.append(distill_loss.item() / len(full_loader))

            x_distill.requires_grad = False
            if self.configs.Trainer.LabelWiseDistillation:
                self.datasets.Distilled.update_label_wise([x_distill, y_distill, n_distill, qid_distill])
            else:
                self.datasets.Distilled.update_query_wise([x_distill, y_distill, n_distill, qid_distill])


            early_stop = self.validate(model_init_itr, "overall")
            if early_stop or (model_init_itr == self.configs.Trainer.ModelInstances - 1):
                self.datasets.Distilled = self.validated_distilled_dataset["overall"]
                wandb_loggers = {
                    "best_step": self.valid_status["overall"]["best_step"],
                    "best_valid_ndcg10": self.valid_status["overall"]["best_metric"]
                }
                self.logging_fn("Exiting early - best_setp: {} - best_valid_ndcg10: {} ..."
                                .format(self.valid_status["overall"]["best_step"],
                                        self.valid_status["overall"]["best_metric"]),
                                True, wandb_loggers)
                break  # If validate returns True -> early stopping

        wandb_loggers = {
            "avg_distill_loss_over_models": sum(models_distill_loss_list) / len(models_distill_loss_list)
        }
        self.logging_fn(None, True, wandb_loggers)

        # After distillation is finished both datasets should be in get_items_query_wise mode, NOTE that
        # label_wise_reading_items attribute in Distilled data is any point during distillation because it will not be
        # used.
        self.datasets.Distilled.get_items_query_wise = True
        self.datasets.TrainFull.get_items_query_wise = True

        for param in list(self.ltr_net.parameters()):
            param.requires_grad = True
        self.ltr_net.init_weights()

        t_distill_end = time()
        t_distill = t_distill_end - t_distill_begin
        msg = "Performing distillation took: {}".format(t_distill)
        wandb_loggers = {
            "t_distill": t_distill
        }
        self.logging_fn(msg, True, wandb_loggers)

        loss_df = pd.DataFrame.from_dict(log_loss, orient="index")
        loss_df = loss_df.transpose()
        save_results(loss_df, os.path.join(self.configs.General.SaveDir, "log_loss.csv"))
