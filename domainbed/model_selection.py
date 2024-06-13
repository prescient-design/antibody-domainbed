# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import numpy as np


def get_test_records(records):
    """Given records with a common test env, get the test records (i.e. the
    records with *only* that single test env and no other test envs)"""
    return records.filter(lambda r: len(r['args']['test_envs']) == 1)


class SelectionMethod:
    """Abstract class whose subclasses implement strategies for model
    selection across hparams and timesteps."""

    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        accs = (records.group('args.hparams_seed')
                .map(lambda _, run_records:
                     (
                         self.run_acc(run_records),
                         run_records
                     )
                     ).filter(lambda x: x[0] is not None)
                .sorted(key=lambda x: x[0]['val_acc'])
                )
        test_env = accs[0][1][0]["args"]["test_envs"][0]
        trial_seed = accs[0][1][0]["args"]["trial_seed"]
        print(
            f"len(accs): {len(accs)} for test_env: {test_env} and trial_seed: {trial_seed} for dataset: {accs[0][1][0]['args']['dataset']} and algorithm: {accs[0][1][0]['args']['algorithm']}")
        return accs[::-1]

    @classmethod
    def hparams_metrics(self, records, metric_str: str):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_metric, records) tuples.
        """
        metrics = (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_metric(run_records, metric_str=metric_str),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0][f'val_{metric_str}'])
        )
        test_env = metrics[0][1][0]["args"]["test_envs"][0]
        trial_seed = metrics[0][1][0]["args"]["trial_seed"]
        print(
            f"len(metrics): {len(metrics)} for test_env: {test_env} and trial_seed: {trial_seed} for dataset: {metrics[0][1][0]['args']['dataset']} and algorithm: {metrics[0][1][0]['args']['algorithm']}")
        return metrics[::-1]

    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None

    @classmethod
    def sweep_metric(self, records, metric_str: str, selection: str = "None"):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_metrics = self.hparams_metrics(records, selection) if selection != "None" else self.hparams_metrics(records, metric_str)
        if len(_hparams_metrics):
            return _hparams_metrics[0][0][f'test_{metric_str}']
        else:
            return None


class OracleSelectionMethod(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all
    checkpoints, we pick the last checkpoint, i.e. no early stopping."""
    name = "test-domain validation set (oracle)"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        test_env = run_records[0]['args']['test_envs'][0]
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': chosen_record[test_in_acc_key],
        }

    @classmethod
    def run_metric(self, run_records, metric_str: str):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        test_env = run_records[0]['args']['test_envs'][0]
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        test_out_prec_key = 'env{}_out_prec'.format(test_env)
        test_in_prec_key = 'env{}_in_prec'.format(test_env)
        test_out_rec_key = 'env{}_out_rec'.format(test_env)
        test_in_rec_key = 'env{}_in_rec'.format(test_env)
        test_out_ece_key = 'env{}_out_ece'.format(test_env)
        test_in_ece_key = 'env{}_in_ece'.format(test_env)
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        try:
            return {
                'val_acc': chosen_record[test_out_acc_key],
                'test_acc': chosen_record[test_in_acc_key],
                'val_prec': chosen_record[test_out_prec_key],
                'test_prec': chosen_record[test_in_prec_key],
                'val_rec': chosen_record[test_out_rec_key],
                'test_rec': chosen_record[test_in_rec_key],
                'val_ece': chosen_record[test_out_ece_key],
                'test_ece': chosen_record[test_in_ece_key]
            }
        except:
            return {
                'val_acc': chosen_record[test_out_acc_key],
                'test_acc': chosen_record[test_in_acc_key]
            }


class IIDAccuracySelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_acc for env in train_envs))"""
    name = "training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        test_env = record['args']['test_envs'][0]
        val_env_keys_acc = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break
            if i != test_env:
                val_env_keys_acc.append(f'env{i}_out_acc')
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        return {
            'val_acc': np.mean([record[key] for key in val_env_keys_acc]),
            'test_acc': record[test_in_acc_key],
        }

    @classmethod
    def _step_metric(self, record, metric_str: str):
        """Given a single record, return a {val_metric, test_metric} dict."""
        test_env = record['args']['test_envs'][0]
        val_env_keys_acc = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break
            if i != test_env:
                val_env_keys_acc.append(f'env{i}_out_acc')
        test_in_acc_key = 'env{}_in_acc'.format(test_env)

        try:
            val_env_keys_prec = []
            for i in itertools.count():
                if f'env{i}_out_prec' not in record:
                    break
                if i != test_env:
                    val_env_keys_prec.append(f'env{i}_out_prec')
            test_in_prec_key = 'env{}_in_prec'.format(test_env)

            val_env_keys_rec = []
            for i in itertools.count():
                if f'env{i}_out_rec' not in record:
                    break
                if i != test_env:
                    val_env_keys_rec.append(f'env{i}_out_rec')
            test_in_rec_key = 'env{}_in_rec'.format(test_env)

            val_env_keys_ece = []
            for i in itertools.count():
                if f'env{i}_out_ece' not in record:
                    break
                if i != test_env:
                    val_env_keys_rec.append(f'env{i}_out_ece')
            test_in_ece_key = 'env{}_in_ece'.format(test_env)

            return {
                'val_acc': np.mean([record[key] for key in val_env_keys_acc]),
                'test_acc': record[test_in_acc_key],
                'val_prec': np.mean([record[key] for key in val_env_keys_prec]),
                'test_prec': record[test_in_prec_key],
                'val_rec': np.mean([record[key] for key in val_env_keys_rec]),
                'test_rec': record[test_in_rec_key],
                'val_ece': np.mean([record[key] for key in val_env_keys_ece]),
                'test_ece': record[test_in_ece_key]
            }
        except:
            return {
                'val_acc': np.mean([record[key] for key in val_env_keys_acc]),
                'test_acc': record[test_in_acc_key]
            }

    @classmethod
    def run_acc(self, run_records):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')

    @classmethod
    def run_metric(self, run_records, metric_str: str):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(lambda x:
            self._step_metric(x, metric_str=metric_str)
            ).argmax(f'val_{metric_str}')


class LeaveOneOutSelectionMethod(SelectionMethod):
    """Picks (hparams, step) by leave-one-out cross validation."""
    name = "leave-one-domain-out cross-validation"

    @classmethod
    def _step_acc(self, records):
        """Return the {val_acc, test_acc} for a group of records corresponding
        to a single step."""
        test_records = get_test_records(records)
        if len(test_records) != 1:
            return None

        test_env = test_records[0]['args']['test_envs'][0]
        n_envs = 0
        for i in itertools.count():
            if f'env{i}_out_acc' not in records[0]:
                break
            n_envs += 1
        val_accs = np.zeros(n_envs) - 1
        for r in records.filter(lambda r: len(r['args']['test_envs']) == 2):
            val_env = (set(r['args']['test_envs']) - set([test_env])).pop()
            val_accs[val_env] = r['env{}_in_acc'.format(val_env)]
        val_accs = list(val_accs[:test_env]) + list(val_accs[test_env+1:])
        if any([v==-1 for v in val_accs]):
            return None
        val_acc = np.sum(val_accs) / (n_envs-1)
        return {
            'val_acc': val_acc,
            'test_acc': test_records[0]['env{}_in_acc'.format(test_env)]
        }

    @classmethod
    def run_acc(self, records):
        step_accs = records.group('step').map(lambda step, step_records:
            self._step_acc(step_records)
        ).filter_not_none()
        if len(step_accs):
            return step_accs.argmax('val_acc')
        else:
            return None


    @classmethod
    def _step_metric(self, records, metric_str: str):
        """Return the {val_metric, test_metric} for a group of records corresponding
        to a single step."""
        test_records = get_test_records(records)
        if len(test_records) != 1:
            return None

        test_env = test_records[0]['args']['test_envs'][0]
        n_envs = 0
        for i in itertools.count():
            if f'env{i}_out_{metric_str}' not in records[0]:
                break
            n_envs += 1
        val_metrics = np.zeros(n_envs) - 1
        for r in records.filter(lambda r: len(r['args']['test_envs']) == 2):
            val_env = (set(r['args']['test_envs']) - set([test_env])).pop()
            val_metrics[val_env] = r[f'env{val_env}_in_{metric_str}']
        val_metrics = list(val_metrics[:test_env]) + list(val_metrics[test_env+1:])
        if any([v==-1 for v in val_metrics]):
            return None
        val_metric = np.sum(val_metrics) / (n_envs-1)
        return {
            f'val_{metric_str}': val_metric,
            f'test_{metric_str}': test_records[0][f'env{test_env}_in_{val_metric}']
        }

    @classmethod
    def run_metric(self, records, metric_str: str):
        step_metrics = records.group('step').map(lambda step, step_records:
            self._step_metric(step_records, metric_str)
        ).filter_not_none()
        if len(step_metrics):
            return step_metrics.argmax(f'val_{metric_str}')
        else:
            return None



