import numpy as np
import matplotlib.pyplot as plt
from ..utils import bins_reliability_multiclass, bins_reliability_binary


def hist_plot_adjust(pad=0.00):
    plt.xlim(-pad, 1 + pad)
    plt.ylim(-pad, 1 + pad)
    plt.xticks(np.linspace(0, 1, 6))
    plt.yticks(np.linspace(0.2, 1.0, 5))
    plt.gca().set_aspect('equal')
    plt.gca().tick_params(length=0)


def reliability_plot(bin_confs, bin_accs, weights,
                     name='reliability plot', acc_label='accuracy',
                     show=False, binary=False):
    '''
    Args:
        bin_confs: np.array (n,), mean confidence for each bin
        bin_accs: np.array (n,), accuracy for each bin
        weights: np.array (n,), normalized number of samples for each bin
        name: str, plot title
        acc_label: str, meaning of bin_accs
        show: bool, if True, plt.show() will be called
    '''
    n_bins = len(bin_confs)
    bins = np.linspace(0, 1, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    plt.bar(centers, bin_confs, color=(1, 0, 0, 0.5), edgecolor='black',
            label='confidence', width=1/n_bins)
    plt.bar(centers, bin_accs, color=(0, 0, 1, 0.5), edgecolor='black',
            label=acc_label, width=1/n_bins)
    plt.bar(centers, weights, color=(0, 1.0, 0.5, 0.8), edgecolor='black',
            label='weight', width=0.5/n_bins)
    plt.plot([0, 1], [0, 1], color='silver', linestyle='--')
    plt.xlabel('confidence')
    plt.legend()
    plt.title(name)
    hist_plot_adjust()
    if show:
        plt.show()


def ReliabilityPlot(true_classes, confs, n_bins=10, binary=False, **kwargs):
    '''
    Wrap for reliability_plot().
    Args:
        true_classes: np.array (n,) of integers in range(0, n_classes)
        confs: np.array (n, n_classes) of predicted probabilities
        n_bins: int, number of bins
        binary: bool, if True, plot binary version
        **kwargs: see reliability_plot()
    '''
    if binary:
        rel = bins_reliability_binary(true_classes, confs, n_bins)
    else:
        rel = bins_reliability_multiclass(true_classes, confs, n_bins)
    reliability_plot(*rel, **kwargs)
