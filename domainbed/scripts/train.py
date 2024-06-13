# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
from functools import partial

import numpy as np
import PIL
import torch
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.lib.collate import rn_collate, esm_collate, pad_x,Alphabet, PaddCollator
from domainbed.lib import create_logger
logger = create_logger(__name__)

import warnings
warnings.filterwarnings('always')


import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default=".")
    parser.add_argument('--dataset', type=str, default="Ab")
    parser.add_argument('--algorithm', type=str, default="IRM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N ste1ps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="./")
    parser.add_argument('--target', type=str, default="None")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--init_step', action='store_true')
    parser.add_argument('--path_for_init', type=str, default="None")
    parser.add_argument('--use_esm', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset, args.use_esm)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, args.use_esm,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Device: ", device)

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](
            args.data_dir,
            args.test_envs, hparams, args.target, use_esm=args.use_esm)
        logger.info("Built dataset")
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []

    for env_i, env in enumerate(dataset):
        uda = []
        logger.info(f"Constructing environment {env_i}")

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))
    logger.info("Done with in/out splits")

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    #############
    # Algorithm #
    #############
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    logger.info(f"Algorithm: {args.algorithm}")
    if args.algorithm == "ERM":
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                    len(dataset) - len(args.test_envs), hparams,
                                    init_step=args.init_step,
                                    path_for_init=args.path_for_init)
    else:
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                    len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)
    algorithm.to(device)

    ###########
    # Loaders #
    ###########
    # Pass in custom collate function from featurizer's own tokenizer if ESM
    if args.use_esm:
        try:
            collate_fn = partial(
                esm_collate, x_collate_fn=algorithm.featurizer.get_batch_tensor_x)
        except:
            collate_fn = partial(
                esm_collate, x_collate_fn=algorithm.network.featurizer.get_batch_tensor_x)
    else:
        collate_fn = rn_collate 
    logger.info(f"Batch size: {hparams['batch_size']}")
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS,
        collate_fn=collate_fn)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]
    logger.info("Set train dataloader")

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS,
        collate_fn=collate_fn)
        for i, (env, env_weights) in enumerate(uda_splits)]
    logger.info("Set uda dataloader")

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS,
        collate_fn=collate_fn)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename, results=None):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        if results is not None:
            save_dict["results"] = results
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    last_results_keys = None
    best_acc = 0
    for step in range(start_step, n_steps):
        logger.debug(f"Update step: {step}")
        step_start_time = time.time()
        minibatches_device = [(x, y)
            for x, y in next(train_minibatches_iterator)]
        if args.use_esm:
            try:
                minibatches_device = pad_x(
                    minibatches_device,
                    padding_idx=algorithm.featurizer.alphabet.padding_idx)
            except:
                minibatches_device = pad_x(
                    minibatches_device,
                    padding_idx=algorithm.network.featurizer.alphabet.padding_idx)
        minibatches_device = [(x.to(device), y.to(device))
            for x, y in minibatches_device]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device) for x, _ in next(uda_minibatches_iterator)]
            if args.use_esm:
                try:
                    uda_device = pad_x(
                        uda_device,
                        padding_idx=algorithm.featurizer.alphabet.padding_idx)
                except:
                    uda_device = pad_x(
                        uda_device,
                        padding_idx=algorithm.network.featurizer.alphabet.padding_idx)
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc
                feats, gt_labels, preds = misc.features(algorithm, loader, weights, device)
                torch.save({'features': feats, 'labels': gt_labels, 'preds': preds},
                    os.path.join(args.output_dir, 'output.pt'))

            ##========
            agg_val_acc, nagg_val_acc = 0, 0
            for name in results.keys():
                if 'acc' in name and 'out' in name and int(name.split('env')[1].split('_')[0]) not in args.test_envs:
                    agg_val_acc += results[name]
                    nagg_val_acc += 1.
            agg_val_acc /= (nagg_val_acc + 1e-9)
            results['agg_val_acc'] = agg_val_acc

            agg_test_acc, nagg_test_acc = 0, 0
            for name in results.keys():
                if 'acc' in name and name !='agg_val_acc' and int(name.split('env')[1].split('_')[0]) in args.test_envs:
                    agg_test_acc += results[name]
                    nagg_test_acc += 1.
            agg_test_acc /= (nagg_test_acc + 1e-9)
            results['agg_test_acc'] = agg_test_acc
            ##========

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

            # Save best model for ensembling
            if best_acc < agg_val_acc:
                logger.info(f'Saving best model... at update step {step}')
                best_acc = agg_val_acc
                save_checkpoint('best_model.pkl', results=json.dumps(results, sort_keys=True))

    if args.init_step:
        algorithm.save_path_for_future_init(args.path_for_init)
    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

    sys.stdout.file.close()
    sys.stderr.file.close()
