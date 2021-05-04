from functools import partial
from sklearn.isotonic import IsotonicRegression

class HistogramBinningBinary:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.thetas = None
        self.bins = np.linspace(0, 1, n_bins + 1)
        self.bins[-1] += 1e-10

    def fit(self, y_confs, y_true):
        '''
        Args:
            y_confs: np.array estimated probabilities of positive class
            y_true: np.array of 0 and 1
        '''
        _, thetas, weights = bins_reliability_binary(y_true, y_confs,
                                                     n_bins=self.n_bins)
#         in case bin is empty, the probability won't be changed
        thetas[weights == 0] = -1
#         in case bin is empty, replace confidence with bin center
#         centers = ((self.bins[:-1] + self.bins[1:]) * 0.5)
#         thetas[weights == 0] = centers[weights == 0]
        self.thetas = thetas
    
    def transform(self, y_confs):
        '''
        Args:
            y_confs: uncalibrated estimated probability of positive class
        Returns:
            y_confs_calib: calibrated probabilities
        '''
        y_confs_calib = self.thetas[np.digitize(y_confs, self.bins) - 1]
        empty_bins = (y_confs_calib < 0)
        y_confs_calib[empty_bins] = y_confs[empty_bins]
        return y_confs_calib

class IRBinary(IsotonicRegression):
    '''
    Isotonic regression wrapper for binary calibration.
    '''
    def __init__(self):
        super().__init__(increasing=True, out_of_bounds='clip',
                         y_min=0.0, y_max=1.0)

class CalibratorOvR:
    def __init__(self, base, **kwargs):
        '''
        Args:
            base: class of binary calibrator
            kwargs: keyword arguments to initialize each calibrator
        '''
        self.base = partial(base, **kwargs)
        self.ovr_calibrators = []
    
    def fit(self, confs, true_classes):
        '''
        Args:
            confs: np.array (n, n_classes) of predicted probabilities (uncalibrated)
            true_classes: np.array (n,) of integers in range(0, n_classes)
        '''
        self.ovr_calibrators = []
        for class_ in range(confs.shape[1]):
            calibrator = self.base()
            calibrator.fit(confs[:, class_], (true_classes == class_).astype(int))
            self.ovr_calibrators.append(calibrator)
    
    def transform(self, confs):
        '''
        Args:
            confs: np.array (n, n_classes) of predicted probabilities
        Returns:
            calbrated_confs: np.array (n, n_classes) of calibrated probabilities
        '''
        cal_confs = np.stack([calibrator.transform(confs[:, class_]) for
            class_, calibrator in enumerate(self.ovr_calibrators)], axis=1)
        cal_confs /= cal_confs.sum(axis=1, keepdims=True)
        return cal_confs

class HistogramBinningMulticlass(CalibratorOvR):
    def __init__(self, n_bins=15):
        super().__init__(HistogramBinningBinary, n_bins=n_bins)

class IsotonicRegressionMulticlass(CalibratorOvR):
    def __init__(self):
        super().__init__(IRBinary)

