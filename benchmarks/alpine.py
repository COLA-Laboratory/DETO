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


class Alpine:
    def __init__(self, params):
        self.n_var = params.n_var
        np.random.seed(params.seed)

        s = np.random.uniform(size=(params.n_step,),low=0.0, high=np.pi / 2)

        self.functions = []
        self.space = []
        for t in range(params.n_step):
            if self.n_var == 1:
                self.functions.append(partial(Alpine.alpine_function, s=s[t]))
            else:
                print(f"Error in Alpine: wrong {self.n_var} n-var")
                exit(-1)

        self.space = ParameterSpace([ContinuousParameter("x", -10, 10)])

    @staticmethod
    def alpine_function(x, s: float = 0.0, output_noise: float = 0.0):
        x = np.asarray(x)
        y = (x * np.sin(x + np.pi + s) + x / 10).reshape(-1, 1)
        y += np.random.normal(loc=0.0, scale=output_noise, size=y.shape)
        return y
