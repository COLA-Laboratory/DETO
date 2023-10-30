import numpy as np
from sklearn.preprocessing import normalize
from functools import partial
from emukit.core import ParameterSpace, ContinuousParameter

from .benchmarkbase import BenchmarkBase


class Movingpeak(BenchmarkBase):
    def __init__(self, params, shift_length=5.0, height_severity=7.0, width_severity=5.0, lam=0.5, n_peak=5, n_step=11,
                 peak_shape='cone'):
        super().__init__()
        np.random.seed(params.seed)
        self.n_var = params.n_var
        self.shift_length = shift_length
        self.height_severity = height_severity
        self.width_severity = width_severity
        self.lam = lam
        self.n_peak = n_peak
        self.var_bound = np.array([[0, 100]] * self.n_var)
        self.height_bound = np.array([[30, 70]] * n_peak)
        self.width_bound = np.array([[1.0, 12.0]] * n_peak)
        self.n_step = n_step
        self.peak_shape = peak_shape

        current_peak = np.random.random(size=(n_peak, self.n_var)) \
                       * np.tile(self.var_bound[:, 1] - self.var_bound[:, 0], (n_peak, 1)) \
                       + np.tile(self.var_bound[:, 0], (n_peak, 1))
        current_width = np.random.random(size=(n_peak,)) \
                        * (self.width_bound[:, 1] - self.width_bound[:, 0]) \
                        + self.width_bound[:, 0]
        current_height = np.random.random(size=(n_peak,)) \
                         * (self.height_bound[:, 1] - self.height_bound[:, 0]) \
                         + self.height_bound[:, 0]
        previous_shift = normalize(np.random.random(size=(n_peak, self.n_var)), axis=1, norm='l2')

        self.peaks = []
        self.widths = []
        self.heights = []

        self.peaks.append(current_peak)
        self.widths.append(current_width)
        self.heights.append(current_height)

        for t in range(1, n_step):
            peak_shift = self.cal_peak_shift(previous_shift)
            width_shift = self.cal_width_shift()
            height_shift = self.cal_height_shift()
            current_peak = current_peak + peak_shift
            current_height = current_height + height_shift.squeeze()
            current_width = current_width + width_shift.squeeze()
            for i in range(self.n_peak):
                self._fix_bound(current_peak[i, :], self.var_bound)
            self._fix_bound(current_width, self.width_bound)
            self._fix_bound(current_height, self.height_bound)
            previous_shift = peak_shift
            self.peaks.append(current_peak)
            self.widths.append(current_width)
            self.heights.append(current_height)
        self.functions = []
        self.space = []
        self.opt_x = []
        self.opt_y = []
        for t in range(self.n_step):
            self.functions.append(partial(Movingpeak.evaluate,
                                          t=t,
                                          peak_shape=self.peak_shape,
                                          peaks=self.peaks,
                                          heights=self.heights,
                                          widths=self.widths,
                                          n_peak=self.n_peak))
            opt_x, opt_y = self.optimal(t=t,
                                        peak_shape=self.peak_shape,
                                        peaks=self.peaks,
                                        heights=self.heights,
                                        widths=self.widths,
                                        n_peak=self.n_peak)
            self.opt_x.append(opt_x)
            self.opt_y.append(opt_y)

        for i in range(self.n_var):
            self.space.append(ContinuousParameter(f"x{i+1}", self.var_bound[i,0], self.var_bound[i,1]))
        self.space = ParameterSpace(self.space)


    def cal_width_shift(self):
        return self.width_severity * np.random.normal(size=(self.n_peak, 1))

    def cal_height_shift(self):
        return self.height_severity * np.random.normal(size=(self.n_peak, 1))

    def cal_peak_shift(self, previous_shift):
        return (1 - self.lam) * self.shift_length * normalize(
            np.random.random(size=(self.n_peak, self.n_var)) - 0.5, axis=1, norm='l2') + self.lam * previous_shift

    @staticmethod
    def optimal(t,peak_shape,peaks,heights,widths,n_peak):
        current_peak = peaks[t]
        current_height = heights[t]
        optimal_x = np.atleast_2d(current_peak[np.argmax(current_height)])
        optimal_y = Movingpeak.evaluate(optimal_x, t,peak_shape,peaks,heights,widths,n_peak)
        return optimal_x, optimal_y

    @staticmethod
    def local_optimal(t,peak_shape,peaks,heights,widths,n_peak):
        optimal_x = np.atleast_2d(peaks[t])
        optimal_y = Movingpeak.evaluate(optimal_x, t,peak_shape,peaks,heights,widths,n_peak)
        return optimal_x, optimal_y

    @staticmethod
    def evaluate(x, t,peak_shape,peaks,heights,widths,n_peak, output_noise: float = 0.0,):
        x = np.atleast_2d(x)
        n_sample = x.shape[0]
        y = np.zeros(shape=(n_sample, 1))
        if peak_shape == "cone":
            peak_function = Movingpeak.peak_function_cone
        elif peak_shape == "sharp":
            peak_function = Movingpeak.peak_function_cone
        elif peak_shape == "hilly":
            peak_function = Movingpeak.peak_function_cone
        else:
            print("Unknown shape, set to cone")
            peak_function = Movingpeak.peak_function_cone
        for i in range(n_sample):
            y[i, 0] = -peak_function(x[i, ], t,peaks,heights,widths,n_peak)
        y += np.random.normal(loc=0.0, scale=output_noise, size=y.shape)
        return y

    @staticmethod
    def peak_function_cone(x, t,peaks,heights,widths,n_peak):
        current_peak = peaks[t]
        current_height = heights[t]
        current_width = widths[t]
        distance = np.linalg.norm(np.tile(x, (n_peak, 1)) - current_peak, axis=1)
        return np.max(current_height - current_width * distance)

    @staticmethod
    def peak_function_sharp(x, t,peaks,heights,widths,n_peak):
        current_peak = peaks[t]
        current_height = heights[t]
        current_width = widths[t]
        distance = np.linalg.norm(np.tile(x, (n_peak, 1)) - current_peak, axis=1)
        return np.max(current_height / (1 + current_width * distance * distance))

    @staticmethod
    def peak_function_hilly(x, t,peaks,heights,widths,n_peak):
        current_peak = peaks[t]
        current_height = heights[t]
        current_width = widths[t]
        distance = np.linalg.norm(np.tile(x, (n_peak, 1)) - current_peak, axis=1)
        return np.max(current_height - current_width * distance * distance - 0.01 * np.sin(20.0 * distance * distance))

    @staticmethod
    def _fix_bound(data, bound):
        for i in range(data.shape[0]):
            if data[i] < bound[i, 0]:
                data[i] = 2 * bound[i, 0] - data[i]
            elif data[i] > bound[i, 1]:
                data[i] = 2 * bound[i, 1] - data[i]
            while data[i] < bound[i, 0] or data[i] > bound[i, 1]:
                data[i] = data[i] * 0.5 + bound[i, 0] * 0.25 + bound[i, 1] * 0.25