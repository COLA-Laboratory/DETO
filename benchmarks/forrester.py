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

from functools import partial
from typing import Tuple, Callable

import numpy as np
from emukit.core import ParameterSpace, ContinuousParameter


class Forrester:
    def __init__(self, params):
        self.n_var = params.n_var
        np.random.seed(params.seed)
        a = np.random.uniform(size=(params.n_step,), low=0.2, high=3.0)

        b = np.random.uniform(size=(params.n_step,), low=-5.0, high=15.0)

        c = np.random.uniform(size=(params.n_step,), low=-5.0, high=5.0)
        self.functions = []
        self.space = []
        for t in range(params.n_step):
            if self.n_var == 1:
                self.functions.append(partial(Forrester.forrester_function, a=a[t], b=b[t], c=c[t]))
            else:
                print(f"Error in Forrester: wrong {self.n_var} n-var")
                exit(-1)
        self.space = ParameterSpace([ContinuousParameter("x", 0, 1)])

    @staticmethod
    def forrester_function(
            x: np.ndarray,
            a: float = 1.0,
            b: float = 0.0,
            c: float = 0.0,
            output_noise: float = 0.0,
    ) -> np.ndarray:
        x = np.asarray(x)
        y_high = np.power(6 * x - 2, 2) * np.sin(12 * x - 4)
        y = (a * y_high + b * (x - 0.5) - c).reshape(-1, 1)
        y += np.random.normal(loc=0.0, scale=output_noise, size=y.shape)
        return y
