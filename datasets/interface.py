import numpy as np
import random
from typing import List
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from pytorchltr.datasets.svmrank.svmrank import SVMRankDataset


def get_per_query_statistics(query_wise_reading_items_list, labels_count=5):
    mean_per_query_per_label_dict = {
        str(i): [] for i in range(labels_count)
    }
    for item in query_wise_reading_items_list:
        features = item[0]
        rels = item[1]
        n_features = features.shape[1]
        qid = item[3]

        labels = set(rels.reshape(-1).tolist())  # This is expected to be ordered list of labels.
        for tl in labels:
            tmp_feat_idx = (rels == tl).nonzero(as_tuple=True)[0].reshape(-1, 1).repeat(1, n_features)
            tmp_feat = torch.gather(features, 0, tmp_feat_idx)
            mean_vec_per_label_per_query = torch.mean(tmp_feat, 0, keepdim=True).squeeze().tolist()
            mean_vec_per_label_per_query.insert(0, qid)
            mean_per_query_per_label_dict[str(tl)].append(mean_vec_per_label_per_query)
    for k, v in mean_per_query_per_label_dict.items():
        mean_per_query_per_label_dict[k] = pd.DataFrame(mean_per_query_per_label_dict[k])
    return mean_per_query_per_label_dict


def ltr_collate_fn(device, req_grad=False, max_list_size=None):
    device = device
    max_list_size = max_list_size
    req_grad = req_grad

    def _collate_fn(batch: List) -> List:  # batch is like: List of [features, rels, n_qd, qid]
        list_size = max([item[1].shape[0] for item in batch])
        if max_list_size is not None:
            raise NotImplementedError("max_list_size implementation in the rest of this function is not ready!")

        num_features = batch[0][0].shape[1]
        out_features = torch.zeros((len(batch), list_size, num_features))
        out_relevance = torch.zeros((len(batch), list_size), dtype=torch.long)
        out_qid = torch.zeros(len(batch), dtype=torch.long)
        out_n = torch.zeros(len(batch), dtype=torch.long)

        # Collate the whole batch
        for item_index, item in enumerate(batch):

            # Generate random indices when we exceed the list_size. NOT IMPLEMENTED
            xs = item[0]
            if xs.shape[0] > list_size:
                raise NotImplementedError(
                    "Number of relevant items for item {} exceeds max_size".format(item_index))

            # Collate features
            if xs.shape[0] > list_size:
                pass
            else:
                out_features[item_index, 0:xs.shape[0], :] = xs

            # Collate relevance
            if xs.shape[0] > list_size:
                pass
            else:
                rel = item[1]
                rel_n = len(item[1])
                out_relevance[item_index, 0:rel_n] = rel

            # Collate qid and n
            out_qid[item_index] = int(item[3])
            out_n[item_index] = min(int(item[2]), list_size)

            out_features = out_features.to(device)
            out_relevance = out_relevance.to(device)
            out_n = out_n.to(device)
            out_qid = out_qid.to(device)

            if req_grad:
                out_features.requires_grad = True

        return [out_features, out_relevance, out_n, out_qid]

    return _collate_fn


class FullDataset(Dataset):

    def __init__(self, configs, **kwargs):
        self.configs = configs
        assert "data" in kwargs.keys()
        data: SVMRankDataset = kwargs["data"]

        self.split = kwargs["split"]
        self.total_vecs = len(data._ys)
        self.query_wise_reading_items = []
        self.label_wise_reading_items = []
        # by default the items in distilled datasets are read query_wise
        self.get_items_query_wise: bool = True

        self.__fill_in_items(data)

    def __fill_in_items(self, data: SVMRankDataset):
        for item in data:
            features = item.features
            n_features = features.shape[1]
            rels = item.relevance
            qid = item.qid
            if self.configs.Datasets.Full.ID == "MSLR30K" and self.configs.Experiment.Name == "gm_sample_queries_lambda_loss":
                if self.split != "test" and len(rels) > self.configs.Datasets.Full.Thr:
                    self.total_vecs -= len(rels)
                    continue

            self.query_wise_reading_items.append([features, rels, features.shape[0], qid])

            labels = set(rels.reshape(-1).tolist())  # This is expected to be ordered list of labels.

            for tl in labels:
                tmp_feat_idx = (rels == tl).nonzero(as_tuple=True)[0].reshape(-1, 1).repeat(1, n_features)
                tmp_labels_idx = (rels == tl).nonzero(as_tuple=True)[0]
                tmp_feat = torch.gather(features, 0, tmp_feat_idx)
                tmp_labels = torch.gather(rels, 0, tmp_labels_idx)

                self.label_wise_reading_items.append([tmp_feat, tmp_labels, tmp_feat.shape[0], qid])

    def __getitem__(self, index) -> List:
        if self.get_items_query_wise:
            return self.query_wise_reading_items[index]
        return self.label_wise_reading_items[index]

    def __len__(self):
        if self.get_items_query_wise:
            return len(self.query_wise_reading_items)
        return len(self.label_wise_reading_items)

    @property
    def indices(self):
        return list(range(len(self.query_wise_reading_items)))


class DistilledDataset(Dataset):

    def __init__(self, configs, **kwargs):
        self.configs = configs
        self.device = self.configs.General.Device
        assert "full_data" in kwargs.keys()
        full_data: FullDataset = kwargs["full_data"]
        self.distilled_query_indices = []
        self.query_wise_reading_items = []
        self.label_wise_reading_items = []
        self.total_vecs = 0

        # by default the items in distilled datasets are read query_wise
        self.get_items_query_wise: bool = True

        # Sample Queries if Needed.
        if self.configs.Datasets.Distilled.SampleQueryWise:
            if self.configs.Datasets.Distilled.QuerySize < 1:
                total_budget = self.configs.Datasets.Distilled.CompressionRatio * full_data.total_vecs
                available_indices = np.arange(0, len(full_data), 1)
                available_indices = list(np.random.permutation(available_indices))
                n_qd_vecs_distilled = 0
                while n_qd_vecs_distilled < total_budget:
                    selected_query = available_indices.pop()
                    n_qd_vecs_distilled += full_data[selected_query][2]
                    self.distilled_query_indices.append(selected_query)
            else:
                num_items = self.configs.Datasets.Distilled.QuerySize
                # Randomly select queries based on the query sampling budget
                self.distilled_query_indices = np.random.permutation(len(full_data))[:num_items]

        else:
            # In case of no sampling, just keep all the queries available
            self.distilled_query_indices = full_data.indices

        for i, index in enumerate(self.distilled_query_indices):
            item = full_data[index]  # item is like [features, rels, n qd pairs, qid]
            features = item[0]
            n_features = features.shape[1]
            rels = item[1]
            qid = item[3]
            nqd = item[2]

            if self.configs.Datasets.Distilled.SampleDocWise and self.configs.General.Alg == "GM":
                total_budget = int(self.configs.Datasets.Distilled.CompressionRatio * nqd)
                available_docids = np.arange(0, nqd, 1)
                available_docids = list(np.random.permutation(available_docids))[:total_budget]
                features = features[available_docids]
                rels = rels[available_docids]
                nqd = total_budget

            if nqd == 0:
                continue
            if self.configs.General.Alg != "DM":
                self.total_vecs += nqd

            labels = set(rels.reshape(-1).tolist())  # This is expected to be ordered list of labels.

            # Initialize label-wise distilled items if needed
            all_tmp_feat_per_query = []
            all_tmp_rels_per_query = []

            distill_only_these_labels = set(labels)  # By default all labels are considered in doc_wise distillation
            if self.configs.Datasets.Distilled.ExplicitRelLabels >= 0:
                assert self.configs.Datasets.Distilled.RandomSelectionBudget > 0
                distill_only_these_labels = [i for i in range(self.configs.Datasets.Distilled.ExplicitRelLabels+1)]

            for tl in labels:
                tmp_feat_idx = (rels == tl).nonzero(as_tuple=True)[0].reshape(-1, 1).repeat(1, n_features)
                tmp_labels_idx = (rels == tl).nonzero(as_tuple=True)[0]
                tmp_feat = torch.gather(features, 0, tmp_feat_idx)
                tmp_labels = torch.gather(rels, 0, tmp_labels_idx)
                # Sample DocWise if needed
                if self.configs.Datasets.Distilled.SampleDocWise:
                    if tl in distill_only_these_labels:
                        if self.configs.Datasets.Distilled.RandomSelectionBudget > 0:
                            len_feat = len(tmp_feat)
                            if len_feat > self.configs.Datasets.Distilled.RandomSelectionBudget:
                                randomly_selected_indices = random.sample(
                                    [i for i in range(len_feat)],
                                    self.configs.Datasets.Distilled.RandomSelectionBudget)
                                tmp_feat = tmp_feat[randomly_selected_indices]
                                tmp_labels = tmp_labels[randomly_selected_indices]
                            else:
                                pass
                        else:
                            if self.configs.General.Alg == "DM":
                                total_budget_dm = min(self.configs.Datasets.Distilled.CompressionRatio,
                                                      tmp_feat.shape[0])
                                available_docids_dm = np.arange(0, tmp_feat.shape[0], 1)
                                available_docids_dm = list(np.random.permutation(available_docids_dm))[:total_budget_dm]
                                tmp_feat = tmp_feat[available_docids_dm].reshape(-1, tmp_feat.shape[1])
                                tmp_labels = tmp_labels[available_docids_dm].reshape(-1)
                                self.total_vecs += tmp_feat.shape[0]

                        if self.configs.Datasets.Distilled.RandInit:
                            tmp_feat = torch.randn_like(tmp_feat)

                    # update features and rels value based on all_tmp*
                    all_tmp_feat_per_query.append(tmp_feat)
                    all_tmp_rels_per_query.append(tmp_labels)
                else:  # in the case of no document sampling, we still need to initialize qd vectors
                    # no init or selection
                    tmp_feat = tmp_feat
                    tmp_labels = tmp_labels

                    # random init
                    if self.configs.Datasets.Distilled.RandInit:
                        tmp_feat = torch.randn_like(tmp_feat)

                    # update features and rels value based on all_tmp*
                    all_tmp_feat_per_query.append(tmp_feat)
                    all_tmp_rels_per_query.append(tmp_labels)

                tmp_labels = tmp_labels.reshape(-1)
                # Create one_label_wise_item
                self.label_wise_reading_items.append([tmp_feat, tmp_labels, tmp_feat.shape[0], qid])
                # self.label_wise_reading_items.append(SVMRankItem(tmp_feat, tmp_labels, tmp_feat.shape[0],
                #                                                  qid, sparse=False))

            # merge all_tmp* to create features and rels, in the case of no doc sampling, it's better
            # to shuffle them too since they are sorted in the previous loop per label.
            features = torch.cat(all_tmp_feat_per_query)
            rels = torch.cat(all_tmp_rels_per_query).reshape(-1)

            shuffle = np.random.permutation(np.arange(0, features.shape[0]))
            features = features[shuffle]
            rels = rels[shuffle]
            # self.query_wise_reading_items.append(SVMRankItem(features, rels, features.shape[0], qid, sparse=False))
            self.query_wise_reading_items.append([features, rels, features.shape[0], qid])

    def __getitem__(self, index):
        if self.get_items_query_wise:
            return self.query_wise_reading_items[index]
        return self.label_wise_reading_items[index]

    def __len__(self):
        if self.get_items_query_wise:
            return len(self.query_wise_reading_items)
        return len(self.label_wise_reading_items)

    def update_query_wise(self, trained_distilled_data_list: list):
        """
        This function updates the query_wise_reading_items based on the trained distilled data in query_wise_mode.
        :param trained_distilled_data_list:
        :return:
        """
        assert len(trained_distilled_data_list[0]) == len(self.query_wise_reading_items)  # Verify there are equal
        # number of queries in them.
        features = trained_distilled_data_list[0]
        n_qd = trained_distilled_data_list[2]

        for i, item in enumerate(self.query_wise_reading_items):
            item[0] = features[i, 0:n_qd[i], :]

    def update_label_wise(self, trained_distilled_data_list: list):
        """
        This function updates the query_wise_reading_items based on the trained distilled data in label_wise_mode.
        :param trained_distilled_data_list:
        :return:
        """
        x_distill, y_distill, n_distill, qid_distill = trained_distilled_data_list

        offsets = torch.hstack([torch.tensor([0], device=self.configs.General.Device), torch.where(
            qid_distill[1:] != qid_distill[:-1])[0] + 1, torch.tensor([len(qid_distill)], device=self.configs.General.Device)])

        for i, item in enumerate(self.query_wise_reading_items):
            item_qid = item[3]
            begin = offsets[i]
            end = offsets[i+1]
            features = torch.zeros_like(item[0])
            rels = torch.zeros_like(item[1])
            ind_feat_start = 0
            for distill_itr in range(begin, end):
                assert item_qid == qid_distill[distill_itr]
                ind_feat_end = ind_feat_start+n_distill[distill_itr]
                features[ind_feat_start:ind_feat_end] = x_distill[distill_itr, 0:n_distill[distill_itr], :]
                rels[ind_feat_start:ind_feat_end] = y_distill[distill_itr, 0:n_distill[distill_itr]]
                ind_feat_start = ind_feat_end

            shuffled_indices = np.random.permutation(np.arange(0, features.shape[0]))
            item[0] = features[shuffled_indices]
            item[1] = rels[shuffled_indices]

    def collate_full_batch(self,):
        if self.get_items_query_wise:
            return ltr_collate_fn(self.configs.General.Device)(self.query_wise_reading_items)
        return ltr_collate_fn(self.configs.General.Device)(self.label_wise_reading_items)


@dataclass
class DatasetInterfaces:
    TrainFull: FullDataset
    ValidFull: FullDataset
    TestFull: FullDataset
    Distilled: DistilledDataset
