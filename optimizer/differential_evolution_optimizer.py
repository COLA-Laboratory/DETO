# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Tuple

import numpy as np

from emukit.core.optimization.acquisition_optimizer import AcquisitionOptimizerBase
from emukit.core.optimization.context_manager import ContextManager
from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from scipy.optimize import differential_evolution

_log = logging.getLogger(__name__)


class DifferentialEvolutionAcquisitionOptimizer(AcquisitionOptimizerBase):

    def __init__(self, space: ParameterSpace, popsize: int = 100) -> None:
        """
        :param space: The parameter space spanning the search problem.
        :param popsize: Number of population for DE.
        """
        super().__init__(space)
        self.popsize = popsize

    def _optimize(self, acquisition: Acquisition, context_manager: ContextManager)\
        -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of abstract method.

        See AcquisitionOptimizerBase._optimizer for parameter descriptions.
        See class docstring for implementation details.
        """
        _log.info("Starting differential evolution optimization of acquisition function {}"
                  .format(type(acquisition)))
        bounds = context_manager.contextfree_space.get_bounds()
        #acquisition_func = acquisition.evaluate
        def acquisiton_func(x0):
            return -acquisition.evaluate(np.atleast_2d(x0)).squeeze()
        res = differential_evolution(acquisiton_func,bounds)
        return np.atleast_2d(res.x), -np.atleast_2d(res.fun)
