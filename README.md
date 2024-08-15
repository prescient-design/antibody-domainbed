# Welcome to Antibody DomainBed

(submission #1590 at NeurIPS 2024 Datasets and Benchmarks Track)

DomainBed is a PyTorch suite containing benchmark datasets and algorithms for domain generalization, as introduced in [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434).

We extend this repo to allow for benchmarking DG algorithms for biological sequences, namely, therapeutic antibodies.
To do so, we adjust the backbones to SeqCNN or ESM, whcih is specified  by adding the `--is_esm` flag to the train script.

## Dataset
Our dataset can be found here:
Before running any tests, please make sure you change the path in domainbed/datasets.py to whereever you store the data

## Quickstart
Set up an environment with all necessary packages `conda create --name <env_name> --file requirements.txt`
Train any DG baseline from Domainbed on the Antibody datset as follows:
`python -m domainbed.scripts.train --dataset AbRosetta --algorithm ERM --output_dir='./some_directory'`

Dataset is available under `domainbed/data/antibody_domainbed_dataset.csv`

All other instructions from the main [Domainbed repo](https://github.com/facebookresearch/DomainBed) hold, please see the original repo for more details on running sweeps.
