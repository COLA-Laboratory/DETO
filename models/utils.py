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

import itertools
from typing import List, Union, Tuple, Dict, Hashable
import numpy as np
import scipy
from GPy.kern import Fixed, BasisFuncKernel, Kern
from GPy.kern.src.stationary import Stationary
from GPy.kern import Coregionalize
from sklearn.cluster import KMeans
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from models import InputData, TaskData


def is_pd(a: np.ndarray) -> bool:
    """Check whether matrix `a` is positive definite via Cholesky decomposition.

    Args:
        a: Input matrix.

    Returns:
        `True` if input matrix is positive-definite, `False` otherwise.
    """

    try:
        _ = np.linalg.cholesky(a)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_pd(a: np.ndarray) -> np.ndarray:
    """Calculate the nearest positive-definite matrix to a given symmetric matrix `a`.

    Nearest is defined by the Frobenius norm.

    Args:
        a: Symmetric matrix. `shape = (n, n)`

    Returns:
        The nearest positive-definite matrix to the input symmetric matrix `a`.
        `shape = (n, n)`
    """
    # compute eigendecomposition of symmetric matrix a
    w, v = np.linalg.eigh(a)

    # account for floating-point accuracy
    spacing = np.spacing(np.linalg.norm(a))

    # clip the eigenvalues at zero
    wp = np.clip(w, spacing, None)

    return np.dot(v, np.dot(np.diag(wp), np.transpose(v)))


def compute_cholesky(matrix: np.ndarray) -> np.ndarray:
    """Calculate the Cholesky decomposition of a matrix.

    If the matrix is singular, a small constant is added to the diagonal of the matrix.
    This method is therefore useful for the calculation of GP posteriors.

    Args:
        matrix: The input matrix. `shape = (n_points, n_points)`

    Returns:
        The Cholesky decomposition stored in the lower triangle.
            `shape = (n_points, n_points)`
    """
    assert len(matrix.shape) <= 2, (
        "The matrix has more than two input dimensions. Cholesky decomposition"
        "impossible."
    )
    assert (
            matrix.shape[0] == matrix.shape[1]
    ), "The matrix is not square. Cholesky decomposition impossible."

    _matrix = np.copy(matrix)  # to avoid modifying the input
    for k in itertools.count(start=1):
        try:
            chol = scipy.linalg.cholesky(_matrix, lower=True)
        except scipy.linalg.LinAlgError:
            # Increase eigenvalues of matrix
            np.fill_diagonal(_matrix, _matrix.diagonal() + 10 ** k * 1e-8)
        else:
            return chol


class FixedKernel(Fixed):
    """Fixed covariance kernel. Serializable version of the Fixed Kernel from `GPy`.

    Serialization is required to initialize a `gpy_adapter` `Model` using this kernel.
    """

    def __init__(
            self,
            input_dim: int,
            covariance_matrix: np.ndarray,
            active_dims: List[int] = None,
            name="PosteriorCov",
    ):
        """Initialize the kernel.

        Args:
            input_dim: Input dimension of the training data.
            covariance_matrix: The fixed covariance matrix.
            active_dims: Active dimensions.
            name: Name of the kernel.
        """
        super(FixedKernel, self).__init__(
            input_dim=input_dim,
            variance=1.0,
            covariance_matrix=covariance_matrix,
            active_dims=active_dims,
            name=name,
        )
        self.variance.fix()

    def to_dict(self) -> dict:
        """Save the kernel as a dictionary."""
        input_dict = super(Fixed, self)._save_to_input_dict()
        input_dict["covariance_matrix"] = self.fixed_K
        input_dict["class"] = "GPy.kern.Fixed"
        input_dict.pop("useGPU")

        return input_dict


def compute_alpha(model: "RBO", x: InputData) -> np.ndarray:
    r"""Calculate the $\alpha(x)$ Woodbury vector used for computing the boosted
    covariance.

    $$
        \alpha(x) = k(x, X)\left(k(\X, \X) +\sigma^2\mathbb 1\right)^{-1}$,
    $$

    where $k$ is the kernel of `model`, $X$ is the training data of `model`, and
    $\sigma$ is the standard deviation of the observational noise.

    Args:
        model: The Gaussian-process model.
        x: The input data. `shape = (n_points, n_features)`

    Returns:
        The $\alpha$ vector. `shape = (n_points, n_training_points)`
    """
    L = model._gpy_model.posterior.woodbury_chol
    X = InputData(model.X)
    k = model.compute_kernel(X, x)
    return scipy.linalg.solve_triangular(
        L.T, scipy.linalg.solve_triangular(L, k, lower=True)
    )


class CrossTaskKernel(BasisFuncKernel):
    """A kernel that is one iff the X-task corresponds to one of the `task_indices`."""

    def __init__(
            self,
            task_indices: Union[Tuple[int, int], int, np.ndarray],
            index_dim: int,
            variance=1.0,
            name="task_domain",
    ):
        super().__init__(
            input_dim=1,
            variance=variance,
            active_dims=(index_dim,),
            ARD=False,
            name=name,
        )
        self.task_indices = np.atleast_2d(np.asarray(task_indices, dtype=int))
        assert self.task_indices.size >= 1, "Need at least one task."

    def _phi(self, X: np.ndarray) -> np.ndarray:
        # atol maps our floats to tasks
        is_domain_task = np.isclose(X, self.task_indices, atol=0.5, rtol=0)
        return is_domain_task.any(axis=-1, keepdims=True)


class SampleAgeingKernel(Stationary):
    def __init__(self, index_dim, variance=1., s2=2.0, lengthscale=None, ARD=False,
                 name='sample_ageing'):
        super(SampleAgeingKernel, self).__init__(input_dim=1, variance=variance,
                                                 lengthscale=lengthscale,
                                                 ARD=ARD,
                                                 active_dims=(index_dim,),
                                                 name=name)
        self.s2 = s2

    def K_of_r(self, r):
        return self.variance * r * self.s2

    def dK_dr(self, r):
        return self.variance * self.s2


def getTVGPUCBKernel(input_dim, num_outputs, kernel, tau=None,name='tv-gp-ucb'):
    K_task = TVGPUCBKernel(1, num_outputs, active_dims=[input_dim], tau=tau, name='tv-gp-ucb')
    K_task[".*tau*"].constrain_fixed(0.6)
    K = kernel.prod(K_task,name=name)
    return K


class TVGPUCBKernel(Coregionalize):
    def __init__(self, input_dim, output_dim, tau=None, active_dims=None, name='tv-gp-ucb'):
        Kern.__init__(self, input_dim, active_dims, name=name)
        self.output_dim = output_dim
        if tau is None:
            tau = 0.5
        self.tau = Param('tau', tau, Logexp())
        self.link_parameters(self.tau)

    def parameters_changed(self):
        self.B = np.ones((self.output_dim,self.output_dim))
        for i in range(self.output_dim):
            for j in range(self.output_dim):
                self.B[i,j] = pow(self.tau,abs(i-j))

    def update_gradients_full(self, dL_dK, X, X2=None):
        self.tau.gradient = 0.0

class KroneckerDeltaKernel(Stationary):
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='kronecker_delta',
                 useGPU=False):
        super().__init__(input_dim, variance, lengthscale, ARD, active_dims, name, useGPU)

    def K_of_r(self, r):
        ret = r == np.zeros_like(r)
        return self.variance * ret.astype(np.int)

    def dK_dr(self, r):
        return self.variance


class SourceSelection:
    @staticmethod
    def the_k_nearest( source_datasets: Dict[Hashable, TaskData], source_models=None, k: int = 3) -> Tuple[Dict[Hashable, TaskData],list]:
        n_source = len(source_datasets)
        data = {}
        selected = list(np.arange(max(0, n_source - k), n_source))
        for (i,idx) in enumerate(selected):
            data[i] = source_datasets[idx]
        return data, selected

    @staticmethod
    def the_k_different( source_datasets: Dict[Hashable, TaskData], source_models:list=None, k: int = 3) -> Tuple[Dict[Hashable, TaskData],list]:
        n_source = len(source_datasets)
        params = np.zeros((len(source_models),len(source_models[0]._gpy_model.param_array)))

        for (i,model) in enumerate(source_models):
            params[i] = source_models[i]._gpy_model.param_array
        if k >= len(source_models):
            selected = list(np.arange(max(0, n_source - k), n_source))
        else:
            selected= SourceSelection.findKBest(params, k, True)[0]
        data = {}
        for (i,idx) in enumerate(selected):
            data[i] = source_datasets[idx]
        return data, selected

    @staticmethod
    def findKBest(X, n_clusters, previous_optima_at0=False):
        n_sample = X.shape[0]
        cluster = KMeans(n_clusters=n_clusters)
        cluster.fit(X)
        all_dis = cluster.transform(X)
        ret = []
        for i in range(n_clusters):
            best_dis = np.inf
            best = -1
            for j in range(n_sample):
                if cluster.labels_[j] == i and all_dis[j, i] < best_dis:
                    best_dis = all_dis[j, i]
                    best = j
            ret.append(best)
        if previous_optima_at0 is True:
            ret[cluster.labels_[0]] = 0
        return ret, X[ret, :]