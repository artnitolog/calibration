import numpy as np
from scipy.special import softmax
from sklearn.metrics import log_loss
from ..utils import bins_reliability_binary, bins_reliability_multiclass


def ECE(bin_confs, bin_accs, weights):
    '''
    Args (returns from bins_reliability):
        bin_confs: np.array (n,), mean confidence for each bin
        bin_accs: np.array (n,), accuracy for each bin
        weights: np.array (n,)

    Returns:
        ece: expected calibration error
    '''
    diffs = np.abs(bin_confs - bin_accs)
    ece = np.average(diffs, weights=weights)
    return ece


def MCE(bin_confs, bin_accs, weights=None):
    '''
    Args (returns from bins_reliability):
        bin_confs: np.array (n,), mean confidence for each bin
        bin_accs: np.array (n,), accuracy for each bin
        weights: np.array (n,), unused

    Returns:
        mce: maximum calibration error
    '''
    diffs = np.abs(bin_confs - bin_accs)
    mce = diffs.max()
    return mce


def BS(true_classes, confs):
    onehot = np.zeros_like(confs)
    onehot[np.arange(len(confs)), true_classes] = 1
    return ((onehot - confs) ** 2).sum(axis=1).mean()


def cwECE(true_classes, confs, n_bins=15):
    '''
    Args:
        true_classes: np.array (n,) of integers in range(0, n_classes)
        confs: np.array (n, n_classes) of predicted probabilities
        n_bins: int, number of bins (for computing binning metrics)
    
    Returns:
        cwece: classwise expected calibration error
    '''
    cweces = [ECE(*bins_reliability_binary(true_classes == j, confs[:, j], n_bins)) 
              for j in range(confs.shape[1])]
    return np.mean(cweces)


def all_metrics(true_classes, confs, n_bins=15, mul=100,
                return_rel=False):
    '''
    Args:
        true_classes: np.array (n,) of integers in range(0, n_classes)
        confs: np.array (n, n_classes) of predicted probabilities
        n_bins: int, number of bins (for computing binning metrics)
        mul: float, multiplier for metrics with values in range [0, 1],
             default is 100
    
    Returns:
        dictionary with metrics:
            'ACC': accuracy times mul,
            'ECE': expected calibration error times mul,
            'MCE': maximum calibration error times mul,
            'cwECE': classwise expected calibration error times mul,
            'NLL': negative log likelihood,
            'BS': Brier score
    '''
    metrics = {}
    metrics['ACC'] = (confs.argmax(axis=1) == true_classes).mean() * mul
    rel = bins_reliability_multiclass(true_classes, confs, n_bins)
    metrics['ECE'] = ECE(*rel) * mul
    metrics['MCE'] = MCE(*rel) * mul
    metrics['cwECE'] = cwECE(true_classes, confs, n_bins) * mul
    metrics['BS'] = BS(true_classes, confs)
    metrics['NLL'] = log_loss(true_classes, confs)
    return metrics
