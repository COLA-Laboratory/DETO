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


class Branin:
    def __init__(self, params):
        self.n_var = params.n_var
        np.random.seed(params.seed)

        a = np.random.uniform(size=(params.n_step,), low=0.5, high=1.5)
        b = np.random.uniform(size=(params.n_step,), low=0.1, high=0.15)
        c = np.random.uniform(size=(params.n_step,), low=1.0, high=2.0)
        r = np.random.uniform(size=(params.n_step,), low=5.0, high=7.0)
        s = np.random.uniform(size=(params.n_step,), low=8.0, high=12.0)
        tau = np.random.uniform(size=(params.n_step,), low=0.03, high=0.05)

        self.functions = []
        self.space = []
        for t in range(params.n_step):
            if self.n_var == 2:
                self.functions.append(partial(Branin.branin_function, a=a[t], b=b[t], c=c[t], r=r[t], s=s[t], t=tau[t]))
            else:
                print(f"Error in Branin: wrong {self.n_var} n-var")
                exit(-1)

        self.space = ParameterSpace(
            [ContinuousParameter("x1", -5.0, 10.0), ContinuousParameter("x2", 0.0, 15.0)]
        )

    @staticmethod
    def branin_function(
        x: np.ndarray,
        a: float = 1.0,
        b: float = 0.1,
        c: float = 1.0,
        r: float = 5.0,
        s: float = 10.0,
        t: float = 0.05,
        output_noise: float = 0.0,
    ) -> np.ndarray:
        x1 = np.array(x[:, 0]).reshape(-1, 1)
        x2 = np.array(x[:, 1]).reshape(-1, 1)
        y = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        y = y.reshape(-1, 1)
        y += np.random.normal(loc=0.0, scale=output_noise, size=y.shape)
        return y