# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Callable, Union
import numpy as np
import scipy.optimize as opt
from time import time
from emukit.core.interfaces import IModel
from emukit.core.optimization import GradientAcquisitionOptimizer, RandomSearchAcquisitionOptimizer
from emukit.core import ParameterSpace
from emukit.bayesian_optimization.acquisitions import (
    NegativeLowerConfidenceBound as UCB,
)
from optimizer.differential_evolution_optimizer import DifferentialEvolutionAcquisitionOptimizer
from models import TaskData, Model
from utils.lhs import lhs


def shgo_minimize(fun: Callable, search_space: ParameterSpace) -> opt.OptimizeResult:
    """Minimize the benchmark with simplicial homology global optimization, SHGO

    Original paper: https://doi.org/10.1007/s10898-018-0645-y

    Parameters
    -----------
    fun
        The function to be minimized.
    search_space
        Fully described search space with valid bounds and a meaningful prior.

    Returns
    --------
    res
        The optimization result represented as a `OptimizeResult` object.
    """

    def objective(x):
        benchmark_value = fun(np.atleast_2d(x), output_noise=0.0)
        return benchmark_value.squeeze()

    bounds = search_space.get_bounds()
    return opt.shgo(objective, bounds=bounds, sampling_method="sobol")


def run_bo(
        experiment_fun: Callable,
        model: Union[Model, IModel],
        space: ParameterSpace,
        t: int,
        num_init: int,
        num_iter: int,
        noiseless_fun: Callable = None,
        exporter=None,
):
    """Runs Bayesian optimization."""
    exporter.timer['fit'] = 0.0
    exporter.timer['af'] = 0.0
    exporter.timer['fe'] = 0.0

    X_new = model.meta_initial(num_init)
    if X_new is None:
        X_new = space.sample_uniform(num_init)
    exporter.timer['fe'] -= time()
    Y_new = experiment_fun(X_new)
    exporter.timer['fe'] += time()
    X, Y = X_new, Y_new
    for i in range(num_iter):


        print('BO iter %d/%d' % (i, num_iter))
        exporter.timer['fit'] -= time()
        model.fit(TaskData(X, Y), optimize=True)
        exporter.timer['fit'] += time()
        # optimize the AF
        exporter.timer['af'] -= time()
        af = UCB(model, beta=np.float64(3.0))
        optimizer = GradientAcquisitionOptimizer(space)

        ''' add plot 1D here'''
        n_sample, n_var = X.shape
        if i == 0:
            cnt = 0
            if model._meta_data is not None:
                meta = model._meta_data
                tmp_X = []
                tmp_Y = []
                cnt = int(np.max(meta[0][:, -1]) + 1)
                cur = 0
                for i in range(len(meta[0])):
                    if meta[0][i, -1] != cur:
                        np.savetxt(f"debug/meta_{cnt}_{cur}.txt", np.hstack([np.vstack(tmp_X), -np.vstack(tmp_Y)]))
                        tmp_X = []
                        tmp_Y = []
                        cur += 1
                    tmp_X.append(meta[0][i, :-1])
                    tmp_Y.append(meta[1][i])
                np.savetxt(f"debug/meta_{cnt}_{cur}.txt", np.hstack([np.vstack(tmp_X), -np.vstack(tmp_Y)]))
            X_test = None
            if n_var == 1:
                size = 1000
                X_test = np.expand_dims(np.linspace(0, 100, size), -1)
            if n_var == 2:
                size = 100
                X_grids = np.meshgrid(np.linspace(0, 100, size),np.linspace(0, 100, size))
                X_test = np.hstack([np.reshape(X_grids[0],(size*size,1)),np.reshape(X_grids[1],(size*size,1))])
                print(X_test.shape)
            Y_pred, Std_pred = model.predict(X_test)
            Y_true = experiment_fun(X_test)
            ac = af.evaluate(X_test)
            np.savetxt(f"debug/train_{cnt}.txt", np.hstack([X, -Y]))
            np.savetxt(f"debug/test_{cnt}.txt",
                       np.hstack([X_test, -Y_true, -Y_pred, -Y_pred - 0.1 * Std_pred, -Y_pred + 0.1 * Std_pred,ac]))
        ''' end of plot '''

        X_new, ac_new = optimizer.optimize(af)
        exporter.timer['af'] += time()
        exporter.timer['fe'] -= time()
        Y_new = experiment_fun(X_new)
        exporter.timer['fe'] += time()
        X = np.append(X, X_new, axis=0)
        Y = np.append(Y, Y_new, axis=0)


    timer = time() - exporter.timer['last']
    exporter.timer['last'] = time()
    exporter.print_log(
        f"{t}th-BO loop is finished ({int(timer)}s {int(exporter.timer['fit'])}/{int(exporter.timer['af'])}/{int(exporter.timer['fe'])}).\n")

    return X, Y
