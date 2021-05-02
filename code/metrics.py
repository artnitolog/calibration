def bins_reliability_binary(y_true, y_confs, n_bins=10):
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

def bins_reliability_multiclass(true_classes, confs, n_bins=10):
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

def _ECE(bin_confs, bin_accs, weights):
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

def _MCE(bin_confs, bin_accs, weights=None):
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

def _hist_plot_adjust():
    pad = 0.00
    plt.xlim(-pad, 1 + pad)
    plt.ylim(-pad, 1 + pad)
    plt.xticks(np.linspace(0, 1, 6))
    plt.yticks(np.linspace(0.2, 1.0, 5))
    plt.gca().set_aspect('equal')
    plt.gca().tick_params(length=0)

def _reliability_plot(bin_confs, bin_accs, weights=None,
                      name='reliability plot', acc_label='bin accuracy',
                      show=True, path=None):
    '''
    Args:
        bin_confs: np.array (n,), mean confidence for each bin
        bin_accs: np.array (n,), accuracy for each bin or class frequencies
        weights: np.array (n,)
        name: str, plot title
        acc_label: str, meaning of bin_accs
        show: bool, if True, plt.show() will be called
        path: str, location to save figure, default is None

    '''
    n_bins = len(bin_confs)
    bins = np.linspace(0, 1, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    plt.bar(centers, bin_confs, color=(1, 0, 0, 0.5), edgecolor='black',
            label='bin confidence', width=1/n_bins)
    plt.bar(centers, bin_accs, color=(0, 0, 1, 0.5), edgecolor='black',
            label=acc_label, width=1/n_bins)
    if weights is not None:
        plt.bar(centers, weights, color=(0, 1.0, 0.5, 0.8), edgecolor='black',
                label='bin weight', width=0.5/n_bins)
    plt.plot([0, 1], [0, 1], color='silver', linestyle='--')
    plt.xlabel('confidence')
    #  plt.ylabel('accuracy') not only...
    plt.legend()
    plt.title(name)
    _hist_plot_adjust()
    if show:
        plt.show()
    if path is not None:
        plt.savefig(path)

