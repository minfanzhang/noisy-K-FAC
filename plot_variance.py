from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import os


_INPUT_DIM = {
    'fmnist': [784],
    'mnist': [784],
    'cifar10': [32, 32, 3],
    'cifar100': [32, 32, 3]
}

_OUTPUT_DIM = {
    'fmnist': 10,
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100
}


# set global settings
def init_plotting():
    plt.rcParams['figure.figsize'] = (15, 5)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.labelsize'] = 1.0 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 0.8 * plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 0.8 * plt.rcParams['font.size']


def main():
    tf.set_random_seed(1231)
    np.random.seed(1231)

    fig_name = 'grad_variance' + '.pdf'
    grad_variance_path1 = 'experiments/grad_check/81train_acc_preFalse'
    grad_variance_path2 = 'experiments/grad_check/81train_acc_preTrue'
    batch_sizes = [1,4,16,32,64,128,256,512,1024]

    grad_variance = []
    for bs in batch_sizes:
        with open(os.path.join(
            grad_variance_path1, str(bs)+'.pkl'), 'rb') as f:
            grad_variance.append(pickle.load(f))
        f.close()

    grad_variance1 = np.array(grad_variance).reshape(
        [len(batch_sizes), 4])

    grad_variance = []
    for bs in batch_sizes:
        with open(os.path.join(
            grad_variance_path2, str(bs)+'.pkl'), 'rb') as f:
            grad_variance.append(pickle.load(f))
        f.close()

    grad_variance2 = np.array(grad_variance).reshape(
        [len(batch_sizes), 4])

    # plotting
    init_plotting()

    f, (ax1, ax2) = plt.subplots(
        1, 2, sharex='col', sharey='row')

    for i in range(4):
        ax1.plot(batch_sizes, grad_variance1[:,i])
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(color='gray', linestyle="--")
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("variance")

    for i in range(4):
        ax2.plot(batch_sizes, grad_variance2[:,i])
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(color='gray', linestyle="--")
    ax2.set_xlabel("batch size")

    #
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15, wspace=0.15)
    plt.savefig(fig_name)


if __name__ == "__main__":
    main()
