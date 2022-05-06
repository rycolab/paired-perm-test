from argparse import ArgumentParser
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab as pl
from cycler import cycler
from itertools import cycle

from arsenal.timer import timers

from pairedpermtest.accuracy import perm_test
from pairedpermtest.accuracy_polymul import perm_test_polymul
from pairedpermtest.monte_carlo import perm_test_acc as perm_test_mc

LENGTH_MEAN = 12
LENGTH_STD = 10

ACC_MEAN = 0.9543
ACC_STD = 0.1116


def gen_non_neg(mean, std):
    val = np.random.normal(mean, std)
    while val < 0:
        val = np.random.normal(mean, std)
    return val


def gen_rate(mean, std):
    val = np.random.normal(mean, std)
    while val < 0 or val > 1:
        val = np.random.normal(mean, std)
    return val


def gen_lengths(N):
    lengths = np.array([gen_non_neg(LENGTH_MEAN, LENGTH_STD) for _ in range(N)])
    lengths = np.ceil(lengths)
    return lengths


def gen_systems_acc(N, mean_offset=1.0, std_offset=1.0):
    system1 = np.array([gen_rate(ACC_MEAN, ACC_STD) for _ in range(N)])
    mean2 = ACC_MEAN * mean_offset
    std2 = ACC_STD * std_offset
    system2 = np.array([gen_rate(mean2, std2) for _ in range(N)])
    lengths = gen_lengths(N)
    system1 = np.floor(system1 * lengths).astype(int)
    system2 = np.floor(system2 * lengths).astype(int)
    return system1, system2, lengths


def evaluate_times(output_file):
    # For  fast compiling
    A, B, lengths = gen_systems_acc(4, 0.97, 1)
    perm_test(A, B)
    perm_test_polymul(A, B)
    perm_test_mc(A, B, 10)
    # Experiment
    T = timers()
    for N in tqdm(range(500, 10001, 500)):
        for _ in range(5):
            A, B, lengths = gen_systems_acc(N, 0.98, 1)
            with T[f"\\texttt{{exact\_perm\_test}}+\\texttt{{convolve\_DP}}"](n=N):
                perm_test(A, B)
            with T[f"\\texttt{{exact\_perm\_test}}+\\texttt{{convolve\_FFT}}"](n=N):
                perm_test_polymul(A, B)
            for K in [5000, 10000, 20000, 40000]:
                with T[f"\\texttt{{monte\_carlo}} ($K={K}$)"](n=N):
                    perm_test_mc(A, B, K)
    with open(output_file, "wb") as f:
        pickle.dump(T, f)
    return T


def get_time_dataframe(T):
    dfs = {}
    for name in [
        f"\\texttt{{exact\_perm\_test}}+\\texttt{{convolve\_DP}}",
        f"\\texttt{{exact\_perm\_test}}+\\texttt{{convolve\_FFT}}",
        f"\\texttt{{monte\_carlo}} ($K=40000$)", f"\\texttt{{monte\_carlo}} ($K=20000$)",
        f"\\texttt{{monte\_carlo}} ($K=10000$)", f"\\texttt{{monte\_carlo}} ($K=5000$)",
    ]:
        timer = T[name]
        x = timer.dataframe()
        means = x.groupby('n').mean()
        dfs[name] = means
    return dfs


lines = ["-", "--", ":", "-.", (0, (3, 5, 1, 5, 1, 5)), (0, (5, 10))]

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=CB_color_cycle)
plt.rc('axes', prop_cycle=cycler(color=CB_color_cycle))

plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 20})
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')


def graph_times(time_df):
    for i, (name, df) in enumerate(time_df.items()):
        X = df.index
        Y = df['timer']
        plt.plot(X, Y, linestyle=lines[i % 6], label=name)

    plt.ylabel("Runtime (seconds)")
    plt.xlabel("Number of Sentences ($N$)")
    plt.legend(bbox_to_anchor=[0, 0.4], loc='lower left', frameon=False)
    plt.ylim(top=6)


    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=4)

    pl.minorticks_on()
    pl.tight_layout(pad=0)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--graph_only", action="store_true")
    args = parser.parse_args()
    if args.graph_only:
        with open(args.output_file, "rb") as f:
            T = pickle.load(f)
    else:
        T = evaluate_times(args.output_file)
    dfs = get_time_dataframe(T)
    graph_times(dfs)
