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
import numpy as np
from typing import List, Tuple, Callable, Dict, Hashable
from functools import partial
from argparse import Namespace
from time import time

from emukit.core import ParameterSpace
from GPy.kern import RBF

from utils.get_arguments import get_params
from utils.infomation_exporter import InformationExporter
from benchmarks import BenchmarkBase

from models import (
    TaskData,
    WrapperBase,
    DETO
)
from bo.run_bo import run_bo
import models, benchmarks


def get_benchmark(
        parameter: Namespace,
) -> Tuple[Tuple[Callable], ParameterSpace, BenchmarkBase]:
    """Create the benchmark object."""
    benchmark = getattr(benchmarks, parameter.problem)(parameter)
    return benchmark.functions, benchmark.space, benchmark


def get_model(
        model_name: str, space: ParameterSpace, source_data: Dict[Hashable, TaskData],
) -> WrapperBase:
    """Create the model object."""
    if len(source_data) == 0:
        model_class = getattr(models, "RBO")
    else:
        model_class = getattr(models, model_name)
    if model_class == MHGP or model_class == SHGP or model_class == BHGP or model_class == SHGP2:
        model = model_class(space.dimensionality)
    elif model_class == TASD:
        kernel = RBF(space.dimensionality + 1)
        model = model_class(kernel=kernel)
    else:
        kernel = RBF(space.dimensionality)
        model = model_class(kernel=kernel)
    model = WrapperBase(model)
    model.meta_fit(source_data)
    return model


def generate_source_data() -> Dict[Hashable, TaskData]:
    source_data = {}
    return source_data


def run_experiment(parameters: Namespace) -> None:
    """The actual experiment code."""
    technique = parameters.algorithm
    benchmark_name = parameters.problem
    np.random.seed(parameters.seed)
    output_noise = parameters.noise
    n_step = parameters.n_step
    inits = [parameters.init_0] + [parameters.init_n] * n_step
    iters = [parameters.iter_0] + [parameters.iter_n] * n_step
    source_data = {}
    X_all = []
    Y_all = []

    exporter = InformationExporter(parameters)
    fs, space, benchmark = get_benchmark(parameters)
    if technique == 'OPT':
        Y_file = open(exporter.folder_name['data'] + 'Y.txt', 'w')
        X_file = open(exporter.folder_name['data'] + 'X.txt', 'w')
        for t in range(n_step):
            for i in range(iters[t] + inits[t]):
                xs = benchmark.opt_x[t].squeeze()
                y = benchmark.opt_y[t].squeeze()
                X_file.write(f"{t}")
                Y_file.write(f"{t}\t{y}\t{y}\n")
                for x in xs:
                    X_file.write(f"\t{x}")
                X_file.write("\n")
        X_file.close()
        Y_file.close()
        return
    for t in range(n_step):
        # Initialize the benchmark and model

        model = get_model(technique, space, source_data)

        # Run n time step with BO and return the regret
        X, Y = run_bo(
            experiment_fun=partial(fs[t], output_noise=output_noise),
            model=model,
            space=space,
            t=t,
            num_iter=iters[t],
            num_init=inits[t],
            noiseless_fun=partial(fs[t], output_noise=0.0),
            exporter=exporter,
        )
        source_data[t] = TaskData(X, Y)
        Y_best = Y.copy()
        for i in range(Y.shape[0]):
            Y_best[i, 0] = np.min(Y[0:i + 1])
        X_all.append(np.hstack((t * np.ones((X.shape[0], 1)), X)))
        Y_all.append(np.hstack((t * np.ones((X.shape[0], 1)), Y, Y_best)))
    timer = time() - exporter.timer['start']
    exporter.print_log(f'All time step finish ({int(timer)}s)')
    xfmt = '%d' + '\t%lf' * parameters.n_var
    yfmt = '%d\t%lf\t%lf'
    np.savetxt(exporter.folder_name['data'] + 'X.txt', np.vstack(X_all), xfmt)
    np.savetxt(exporter.folder_name['data'] + 'Y.txt', np.vstack(Y_all), yfmt)


if __name__ == "__main__":
    params = get_params()
    run_experiment(params)
