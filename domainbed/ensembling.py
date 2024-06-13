# Copyright (c) Salesforce and its affiliates. All Rights Reserved
import os
import argparse
import time
import json
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from domainbed import networks
from domainbed import datasets
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.lib.collate import esm_collate
from domainbed.lib.reporting import load_records
from domainbed.networks import ESMModel
from domainbed.lib import create_logger
from domainbed.lib.query import Q
logger = create_logger(__name__)


class Algorithm(torch.nn.Module):
    def __init__(self, input_shape, hparams, num_classes, device):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.network.featurizer = networks.Featurizer(input_shape, hparams)
        self.network.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])

        self.net = nn.Sequential(self.featurizer, self.classifier)

        self.featurizer_mo = networks.Featurizer(input_shape, hparams)
        self.classifier_mo = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])

        self.network = self.network.to(device)
        self.network = torch.nn.parallel.DataParallel(self.network).cuda()

        self.network_sma = nn.Sequential(self.featurizer_mo, self.classifier_mo)
        self.network_sma = self.network_sma.to(device)
        self.network_sma = torch.nn.parallel.DataParallel(self.network_sma).cuda()

        self.device = device

    def predict(self, x):
        if self.hparams['algorithm'] == "ERM_SMA":
            output = self.network_sma(x)
        else:
            output = self.network(x)
        return output


def accuracy(models, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            x1, y = data[0], data[-1]
            x = x1.to(device)
            y = y.to(device)
            p = None
            for model in models:
                model.eval()
                p_i = model.predict(x).detach()
                if p is None:
                    p = p_i
                else:
                    p += p_i
            batch_weights = torch.ones(len(x)).cuda()
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y.argmax(1)).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    return correct / total


def precision_recall(models, loader, device):
    ys = torch.Tensor()
    preds = torch.Tensor()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = None
            for model in models:
                model.eval()
                p_i = model.predict(x).detach()
                if p is None:
                    p = p_i
                else:
                    p += p_i
                break
            ys = torch.cat((ys, y.argmax(1).cpu()))
            preds = torch.cat((preds, p.cpu().argmax(1)))
    rec = metrics.recall_score(ys.numpy(), preds.numpy(), zero_division=0)
    prec = metrics.precision_score(ys.numpy(), preds.numpy(), zero_division=0)
    return prec, rec


def rename_dict(D):
    dnew = {}
    for key, val in D.items():
        pre = key.split('.')[0]
        if pre == 'network':
            knew = '.'.join(['network.module'] + key.split('.')[1:])
        else:
            knew = '.'.join(['network_sma.module'] + key.split('.')[1:])
        dnew[knew] = val
    return dnew


def get_test_env_id(path):
    results_path = os.path.join(path, "results.jsonl")
    with open(results_path, "r") as f:
        for j, line in enumerate(f):
            r = json.loads(line[:-1])
            env_id = r['args']['test_envs'][0]
            break
    return env_id


def get_valid_model_selection_paths(path, nenv):
    valid_model_id = [[] for _ in range(nenv)]
    for env in range(nenv):
        cnt = 0
        for i, subdir in enumerate(os.listdir(path)):
            if '.' not in subdir and "done" in os.listdir(os.path.join(path, subdir)):
                test_env_id = get_test_env_id(os.path.join(path, subdir))
                if env == test_env_id:
                    cnt += 1
                    valid_model_id[env].append(f'{path}/{subdir}/best_model.pkl')
    return valid_model_id


def get_dict_folder_to_score(inf_args, trial_seed):
    output_folders = [
        os.path.join(output_dir, path)
        for output_dir in inf_args.output_dir.split(",")
        for path in os.listdir(output_dir)
    ]
    output_folders = [
        output_folder for output_folder in output_folders
        if os.path.isdir(output_folder) and "done" in os.listdir(output_folder) and "best_model.pkl" in os.listdir(output_folder)
    ]

    dict_folder_to_score = {}
    for folder in output_folders:
        model_path = os.path.join(folder, "best_model.pkl") if not inf_args.is_test_domain else os.path.join(folder, "model.pkl")
        save_dict = torch.load(model_path)
        train_args = save_dict["args"]

        if train_args["dataset"] != inf_args.dataset:
            continue
        if train_args["test_envs"] != [inf_args.test_env]:
            continue
        if train_args["algorithm"] != inf_args.algorithm:
            continue
        if train_args["trial_seed"] != trial_seed and trial_seed != -1:
            continue
        if inf_args.is_test_domain:
            results_path = os.path.join(folder, "results.jsonl")
            records = []
            with open(results_path, "r") as f:
                for line in f:
                    records.append(line[:-1])
            loaded_records = records[-1]
        else:
            loaded_records = save_dict["results"]
        score = misc.get_score(
            json.loads(loaded_records),
            [inf_args.test_env],
            is_test_domain=inf_args.is_test_domain)

        print(f"Found: {folder} for trial {trial_seed} with score: {score}")
        dict_folder_to_score[folder] = score

    if len(dict_folder_to_score) == 0:
        raise ValueError(f"No folders found for: {inf_args}")
    return dict_folder_to_score


def get_ensemble_test_acc(hparams, args, device):
    test_acc = {}
    good_checkpoints = []
    for trial_seed in range(0, 3):  # keep the best checkpoint per trial_seed
        dict_folder_to_score = get_dict_folder_to_score(args, trial_seed)
        sorted_checkpoints = sorted(dict_folder_to_score.keys(), key=lambda x: dict_folder_to_score[x], reverse=True)
        good_checkpoints.append(sorted_checkpoints[0])
    print(f"good_checkpoints: {good_checkpoints}")

    dataset = vars(datasets)[args.dataset](
        args.data_dir, [args.test_env], hparams, use_esm=args.use_esm)
    test_acc[args.test_env] = None
    if args.use_esm:
        collate_fn = partial(
            esm_collate, x_collate_fn=ESMModel().get_batch_tensor_x)
    else:
        collate_fn = None
    data_loader = FastDataLoader(
        dataset=dataset[args.test_env],
        batch_size=hparams['batch_size'],  # 64*12
        num_workers=hparams['num_workers'],  # 64
        collate_fn=collate_fn)

    Algorithm_all = []
    input_shape = "use_esm" if args.use_esm else dataset.input_shape
    for model_path in good_checkpoints:
        Algorithm_ = Algorithm(input_shape, hparams, dataset.num_classes, device)
        algorithm_dict = torch.load(os.path.join(model_path, "best_model.pkl")) if not args.is_test_domain else torch.load(os.path.join(model_path, "model.pkl"))
        print(algorithm_dict.keys())
        D = rename_dict(algorithm_dict['model_dict'])
        Algorithm_.load_state_dict(D, strict=False)
        Algorithm_all.append(Algorithm_)
    print(f'Test Domain: {dataset.ENVIRONMENTS[args.test_env]} / Size ensemble: {len(Algorithm_all)}')
    acc = accuracy(Algorithm_all, data_loader, device)
    print(f'  Test domain Acc: {100.*acc:.2f}%')
    prec, rec = precision_recall(Algorithm_all, data_loader, device)
    print(f'  Test domain Prec: {100. * prec:.2f}% / Rec: {100. * rec:.2f}%')
    test_acc[args.test_env] = acc

    return test_acc


parser = argparse.ArgumentParser(description='Ensemble of Averages')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--dataset', type=str, default="Ab8")
parser.add_argument('--test_env', type=int, default=7)
parser.add_argument('--output_dir', type=str, help='the experiment directory where the results of domainbed.scripts.sweep were saved')
parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
parser.add_argument("--algorithm", type=str, default="ERM")
parser.add_argument("--use_esm", action="store_true")
parser.add_argument("--is_test_domain", action="store_true")
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Device: ", device)

hparams = {
    'data_augmentation': False, "nonlinear_classifier": False,
    "resnet_dropout": 0, "batch_size": 128, "num_workers": 1,
    "algorithm": args.algorithm, "add_noise": False}
if args.hparams:
    hparams.update(json.loads(args.hparams))

tic = time.time()
test_acc = get_ensemble_test_acc(hparams, args, device)
test_acc = {k: float(f'{100.*test_acc[k]:.1f}') for k in test_acc.keys()}
toc = time.time()
print(
    f'Avg: {np.array(list(test_acc.values())).mean():.1f}, Time taken: {toc-tic:.2f}s')
