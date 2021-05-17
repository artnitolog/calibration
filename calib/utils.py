import numpy as np


def bins_reliability_binary(y_true, y_confs, n_bins=15):
    '''
    Args:
        y_true: np.array (n,) of 0 and 1, real classes
        y_confs: np.array (n,), predicted probabilities of positive class
        n_bins: int, number of bins
    
    Returns:
        bin_confs: np.array (n,), mean confidence for each bin
        bin_accs: np.array (n,), frequency of positives for each bin
        weights: np.array (n,), normalized number of samples for each bin
    '''
    bins = np.linspace(0, 1, n_bins + 1)
    # [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0 + eps)
    bins[-1] = 1 + 1e-10
    # find which bin each sample is assigned to
    bin_inds = np.digitize(y_confs, bins, right=False) - 1
    # count number of samples in each bin
    total = np.bincount(bin_inds, minlength=n_bins)
    # find mean confidence for each bin
    bin_confs = np.bincount(bin_inds, y_confs, minlength=n_bins)
    np.divide(bin_confs, total, out=bin_confs, where=total!=0)
    # find accuracy for each bin
    bin_accs = np.bincount(bin_inds, y_true, minlength=n_bins)
    np.divide(bin_accs, total, out=bin_accs, where=total!=0)
    weights = total / total.sum()
    return bin_confs, bin_accs, weights


def bins_reliability_multiclass(true_classes, confs, n_bins=15):
    '''
    Args:
        true_classes: np.array (n,) of integers in range(0, n_classes)
        confs: np.array (n, n_classes) of predicted probabilities
        n_bins: int, number of bins
    
    Returns:
        bin_confs: np.array (n,), mean confidence for each bin
        bin_accs: np.array (n,), accuracy for each bin
        weights: np.array (n,), normalized number of samples for each bin
    '''
    is_correct = (true_classes == confs.argmax(axis=1))
    prediction_confs = confs.max(axis=1)
    return bins_reliability_binary(is_correct, prediction_confs, n_bins) 


#def tt(array_):
    #return torch.from_numpy(array_)


#def tnp(tensor_):
    #return tensor_.detach().cpu().numpy()
