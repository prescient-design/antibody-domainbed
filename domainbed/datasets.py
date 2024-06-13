# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import TensorDataset, Dataset, Subset
ImageFile.LOAD_TRUNCATED_IMAGES = True
from domainbed.lib import create_logger
logger = create_logger(__name__, level="debug")

DATASETS = ["AbRosetta"]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class SingleDomainAbDataset(Dataset):
    def __init__(self, df, max_target_len: int):
        self.df = df.reset_index(drop=True)
        self.max_target_len = max_target_len
        self.aa_to_int = dict(zip(
            ['M', 'N', 'P', 'Y', 'D',
             'H', 'Q', 'C', 'K', 'G',
             'R', 'S', 'E', 'I', 'L',
             'T', 'F', 'V', 'A', 'W',
             '-', '.'], range(22)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        try:
            onehot_heavy = F.one_hot(
                torch.tensor(
                    [self.aa_to_int[residue] for residue in row['heavy_chain_full_aho']]),
                num_classes=22)
            onehot_light = F.one_hot(
                torch.tensor(
                    [self.aa_to_int[residue] for residue in row['light_chain_full_aho']]),
                num_classes=22)
            # Pad antigen
            onehot_target = F.one_hot(
                torch.tensor(
                    [self.aa_to_int[residue] for residue in row['target_seq']]),
                num_classes=22
            )  # [len_target, 22]
        except:
            onehot_heavy = F.one_hot(
                torch.tensor(
                    [self.aa_to_int[residue] for residue in row['fv_heavy_aho']]),
                num_classes=22)
            onehot_light = F.one_hot(
                torch.tensor(
                    [self.aa_to_int[residue] for residue in row['fv_light_aho']]),
                num_classes=22)
            # Pad antigen
            onehot_target = F.one_hot(
                torch.tensor(
                    [self.aa_to_int[residue] for residue in row['ag_wt']]),
                num_classes=22
            )  # [len_target, 22]
        onehot_target = F.pad(
            onehot_target, pad=(0, 0, 0, self.max_target_len - onehot_target.shape[0]), value=22)
        onehot_light = F.pad(
            onehot_light, pad=(0, 0, 0, 149 - onehot_light.shape[0]), value=22)
        ab = torch.cat([onehot_heavy, onehot_light], dim=0)  # [149+149, 22]
        x = torch.cat([ab, onehot_target], dim=0).float()  # [149+149+self.max_target_len, 22]

        y = F.one_hot(torch.tensor([int(row['is_binder'])]), num_classes=2).reshape(-1).float()
        return x.reshape(-1, 22), y


class ESMSingleDomainAbDataset(Dataset):
    def __init__(
            self,
            df,
            max_target_len: int
            ):
        self.df = df.reset_index(drop=True)
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        try:
            heavy = row['heavy_chain_full_aho']
            light = row['light_chain_full_aho']
            target = row['target_seq'][:self.max_target_len]
        except:
            heavy = row['fv_heavy_aho']
            light = row['fv_light_aho']
            target = row['ag_wt'][:self.max_target_len]
        y = F.one_hot(
            torch.tensor([int(row['is_binder'])]), num_classes=2).reshape(-1).float()
        polyglycine_linker = "G"*25
        x_cat = heavy + polyglycine_linker + light + polyglycine_linker + target  # TODO jwp: confirm separation
        return ("", x_cat), y  # for compatibility with esm batch loader


class AbRosetta(MultipleDomainDataset):
    N_STEPS = 15000           # Default, subclasses may override
    CHECKPOINT_FREQ = 500    # Default, subclasses may override
    N_WORKERS = 1            # Default, subclasses may override
    ENVIRONMENTS = [f'round_{n}' for n in range(0, 4)]      # Subclasses should override

    def __init__(
            self, root=None, test_envs=None, hparams=None, target='None',
            use_esm=False, max_target_len=None):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []

        try:
            df = pd.read_csv("./data/antibody_domainbed_dataset.csv")
        except:
            print("Error loading the dataset, was it downloaded?")
            return
        df.dropna(subset=['ddG'], inplace=True)
        df.drop_duplicates(subset=['fv_heavy_aho', 'fv_light_aho', 'fv_light_aho_seed', 'fv_heavy_aho_seed', 'ddG'],
                           inplace=True, keep='first')
        df.loc[df['target'] == 'IL-6', 'target'] = 'IL6'
        df.dropna(subset=['ddG'], inplace=True)
        df.drop_duplicates(subset=['seqid'], inplace=True, keep='last')
        print(df.groupby(['env', 'target'])['target'].count())

        df['is_binder'] = df['ddG'].apply(
            lambda x: bool(x < 0.0))

        # Max target length
        if max_target_len is None:
            target_len = df['ag_wt'].apply(len)
            max_target_len = target_len.max()
        self.max_target_len = max_target_len
        logger.info(f"Max target length: {self.max_target_len}")
        self.df = df.reset_index(drop=True)
        if use_esm:
            single_domain_dataset_class = globals()['ESMSingleDomainAbDataset']
            self.input_shape = 'use_esm'
        else:
            single_domain_dataset_class = globals()['SingleDomainAbDataset']
            self.input_shape = (149+149+self.max_target_len, 22)

        # Some basic checks
        assert np.all(
            self.df['fv_heavy_aho'].apply(len).values == 149)
        assert np.all(
            np.isin(self.df['fv_light_aho'].apply(len).values, [148, 149]))

        print(df.groupby(['env', 'target', 'is_binder'])['is_binder'].count())
        self.df = df
        print(df.shape)
        self.ENVIRONMENTS = np.array(sorted(df['env'].unique()))
        print(self.ENVIRONMENTS)
        df.to_csv('dataset_w_meta.csv')
        for round_str in self.ENVIRONMENTS:
            self.datasets.append(
                single_domain_dataset_class(
                    self.df[self.df['env'] == round_str],
                    self.max_target_len)
                )
