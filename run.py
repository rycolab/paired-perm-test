from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from time import time

from pairedpermtest.monte_carlo import perm_test_acc as mc_acc, perm_test_f1 as mc_f1
from pairedpermtest.accuracy_polymul import perm_test_polymul as acc
from pairedpermtest.f1 import perm_test as f1, f1state


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument('--metric', required=True, choices=['acc', 'F1'])
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--MC', default=0, type=int)
    parser.add_argument('--delim', choices=[',', 'tab'], default=',')
    return parser.parse_args()


def _read_acc_file(file, delim=','):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(delim) for line in lines]
    N = len(lines)
    xs = np.zeros(N, dtype=int)
    ys = np.zeros(N, dtype=int)
    for i, (x, y) in enumerate(lines):
        xs[i] = int(x.strip())
        ys[i] = int(y.strip())
    return xs, ys


def _read_f1_file(file, delim=','):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(delim) for line in lines]
    xs = []
    ys = []
    for x_tp, x_i, y_tp, y_i in lines:
        xs.append(f1state(int(x_tp.strip()), int(x_i.strip())))
        ys.append(f1state(int(y_tp.strip()), int(y_i.strip())))
    return xs, ys


def main():
    args = _parse_args()
    delim = '\t' if args.delim == 'tab' else args.delim

    print(f"Reading file: {args.input.stem}")
    if args.metric == "acc":
        perm_test = acc
        mc_test = mc_acc
        xs, ys = _read_acc_file(args.input, delim)
        metric = "accuracy"
    else: # args.metric == "f1"
        perm_test = f1
        mc_test = mc_f1
        xs, ys = _read_f1_file(args.input, delim)
        metric = "F1 score"

    if args.MC:
        print(f"Running approximate MC paired-permutation test on difference of {metric} using {args.MC} samples")
        start = time()
        pvalue = mc_test(xs, ys, args.MC)
        total_time = time() - start
        print(f"p-value: {pvalue}")
        print(f"Test took {total_time:.4f} seconds")
    else:
        print(f"Running exact paired-permutation test on difference of {metric}")
        if metric == "F1 score" and len(xs) > 500:
            print(f"WARNING: The exact test for F1 scores is slow for large inputs"
                  f"we recommend aborting call as you have N={len(xs)}")
        start = time()
        pvalue = perm_test(xs, ys)
        total_time = time() - start
        print(f"p-value: {pvalue}")
        print(f"Test took {total_time:.4f} seconds")


if __name__ == '__main__':
    main()
