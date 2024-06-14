# ictir-2024

In order to run experiments, first install required packages using `pip install -r requirements.txt`. Then, you need to configure the config files in `configs/` directory. Most importantly, you need to specify the location of the dataset you are running experiments with.

Config files currently are adjusted in the same way we ran our experiments. You can find all the experiments we ran in `jobs/` directory including hyperparameter files with all configurations. We have ran three main experiments, query-wise and doc-wise gradient matching distillation and label-wise distribution matching distillation.

If `rand_init` in the configuration is specified as `True`, then the distilled dataset will be randomly initilized at the beginning, else, distilled dataset will be initialized using randomly sampled dataset.

We used WANDB for running experiments, every metric and training parameters of each completed experiments will be stored in a unique directory. You can use `results_analysis.ipynb` to plot the figures and apply the significance testing.
