from argparse import ArgumentParser
import numpy as np


def get_params():
    """

    """
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--n-var', type=int, default=3,
                        help='dimension of input space')
    parser.add_argument('--n-obj', type=int, default=1,
                        help='dimension of output space')
    parser.add_argument('--init-0', type=int, default=None,
                        help='number of initial sample at time step 0')
    parser.add_argument('--iter-0', type=int, default=None,
                        help='number of FEs at time step 0')
    parser.add_argument('--init-n', type=int, default=None,
                        help='number of initial sample at time step n')
    parser.add_argument('--iter-n', type=int, default=None,
                        help='number of FEs at time step n')
    parser.add_argument('--n-step', type=int, default=3,
                        help='number of time step')
    parser.add_argument('--algorithm', type=str, default='RBO',
                        help='algorithm name')
    parser.add_argument('-p', '--problem', type=str, default='Movingpeak',
                        help='optimization problem')
    parser.add_argument('--algorithm-extend', type=str, default=None,
                        help='custom algorithm name to distinguish between experiments on same '
                             'algorithm')
    parser.add_argument('--problem-extend', type=str, default=None,
                        help='custom problem name to distinguish between experiments on same '
                             'problem')
    parser.add_argument('--approximation', type=str, default=None,
                        help='kernel approximation methods')
    parser.add_argument('--noise',type=float,default=0.0,
                        help='output noise for the benchmark problem')
    params, _ = parser.parse_known_args(None)

    # Setup default args
    params.init_0 = (params.n_var * 11 - 1) if params.init_0 is None else params.init_0
    params.iter_0 = (params.n_var * 11 - 1) if params.iter_0 is None else params.iter_0
    if params.algorithm in ['HGP','MTGP','WSGP','SHGP','DIN','CBO','TASD','MHGP','BHGP','MHGP','SHGP2','RGPE','HGP2','HGP3','HGP4','TVGPUCB','PBO','TVGPUCB2','MTGP_LMC']:
        params.init_n = 2*params.n_var if params.init_n is None else params.init_n
        params.iter_n = params.n_var * 5 if params.iter_n is None else params.iter_n
    else:
        params.init_n = params.n_var * 5 if params.init_n is None else params.init_n
        params.iter_n = 2*params.n_var if params.iter_n is None else params.iter_n
    return params
