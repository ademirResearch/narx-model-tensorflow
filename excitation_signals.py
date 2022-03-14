from scipy import signal
import numpy as np


class FilteredGaussianWhiteNoise:
    def __init__(self, ) -> None:
        self.b, self.a = signal.butter(2, 0.05)
        self.filtered_signal = None
        pass

    def get_signal(self, samples, mean, std):
        """Computes filtered white noise signal. 1D with length=samples

        Args:
            samples (_type_): Number of points for the signal 1D
            mean (_type_): Normal distribution mean value
            std (_type_): Normal distribution standard deviation value

        Returns:
            _type_: Filtered signal. Parameters a, and b belong to the butterworth filter (Hardcoded at init this class)
        """
        # Generate white noise signal
        _raw_signal = np.random.normal(mean, std, size=samples)
        # Apply filter
        self.filtered_signal = signal.filtfilt(self.b, self.a, _raw_signal)
        return self.filtered_signal



