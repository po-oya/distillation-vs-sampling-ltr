import os
from pytorchltr.datasets.svmrank.svmrank import  SVMRankDataset


class Yahoo(SVMRankDataset):
    """
    Utility class for using the Yahoo dataset:
    """

    splits = {
        "train": "train.txt",
        "test": "test.txt",
        "vali": "vali.txt"
    }

    def __init__(self, location: str,
                 split: str = "train", normalize: bool = False,
                 filter_queries: bool = None):
        """
        Args:
            location: Directory where the dataset is located.
            split: The data split to load ("train", "test" or "vali")
            normalize: Whether to perform query-level feature
                normalization.
            filter_queries: Whether to filter out queries that
                have no relevant items. If not given this will filter queries
                for the test set but not the train set.
        """
        # Check if specified split and fold exists.
        if split not in Yahoo.splits.keys():
            raise ValueError("unrecognized data split '%s'" % str(split))

        # Only filter queries on non-train splits.
        if filter_queries is None:
            filter_queries = False if split == "train" else True

        # Initialize the dataset.
        datafile = os.path.join(location, Yahoo.splits[split])
        super().__init__(file=datafile, sparse=False, normalize=normalize,
                         filter_queries=filter_queries, zero_based="auto")