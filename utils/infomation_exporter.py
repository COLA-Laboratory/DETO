import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from time import time


class InformationExporter:
    def __init__(self,args):
        self.folder_name = {}
        algorithm_name = f'{args.algorithm}' if args.algorithm_extend is None \
            else f'{args.algorithm}-{args.algorithm_extend}'
        problem_name = f'{args.problem}' if args.problem_extend is None \
            else f'{args.problem}-{args.problem_extend}'
        self.folder_name['fig'] = f'output/figs/{problem_name}/x{args.n_var}y{args.n_obj}/' \
                                  + '/' + algorithm_name + '/' + f'{args.seed}' + '/'
        self.folder_name['data'] = f'output/data/{problem_name}/x{args.n_var}y{args.n_obj}/' \
                                   + '/' + algorithm_name + '/' + f'{args.seed}' + '/'
        self.folder_name['debug'] = f'output/debug/{problem_name}/x{args.n_var}y{args.n_obj}/' \
                                    + '/' + algorithm_name + '/' + f'{args.seed}' + '/'
        self.folder_name['log'] = f'output/log/{problem_name}/x{args.n_var}y{args.n_obj}/' \
                                  + '/' + algorithm_name + '/' + f'{args.seed}' + '/'
        #os.makedirs(self.folder_name['fig'], exist_ok=True)
        os.makedirs(self.folder_name['data'], exist_ok=True)
        #os.makedirs(self.folder_name['debug'], exist_ok=True)
        os.makedirs(self.folder_name['log'], exist_ok=True)
        self.log_file = open(self.folder_name['log']+'/log.txt','w')
        self.timer = {}
        self.timer['start'] = time()
        self.timer['last'] = time()

    def print_log(self,info):
        self.log_file.write(info)
        self.log_file.flush()

    @staticmethod
    def fill2D(X, Y1, Y2, c='gray', ax=None, file=None, show=False):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.fill_between(X.squeeze(), Y1.squeeze(), Y2.squeeze(), color=c, alpha=0.5)
        if file is not None:
            plt.grid()
            plt.savefig(file, format='pdf')
        if show:
            plt.grid()
            plt.show()
        if file is None and show is False:
            return ax
        return None

    @staticmethod
    def contour2D(X_grid, Y_grid, cmap=cm.Blues, ax=None, file=None, show=False, label=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.set_xlabel('$x_1$', fontsize=13)
        ax.set_ylabel('$x_2$', fontsize=13)
        ax.contour(X_grid, X_grid.T, Y_grid, cmap=cmap, label=label)
        if file is not None:
            plt.grid()
            plt.savefig(file, format='pdf')
        if show:
            plt.grid()
            plt.show()
        if file is None and show is False:
            return ax
        return None

    @staticmethod
    def plot2D(X, Y, c='black', ls='', marker='o', fillstyle=None, label=None, ax=None, file=None, show=False,
               show_legend=False):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.plot(X, Y, c=c, ls=ls, marker=marker, label=label, fillstyle=fillstyle)
        ax.set_xlabel('$\mathbf{x}$', fontsize=13)
        ax.set_ylabel('$\mathbf{x}$', fontsize=13)
        ax.tick_params(axis='both', labelsize=13)

        if file is not None:
            if show_legend:
                plt.legend()
            plt.grid()
            plt.savefig(file, format='pdf')
        if show:
            if show_legend:
                plt.legend()
            plt.grid()
            plt.show()
        if file is None and show is False:
            return ax
        return None

    @staticmethod
    def surface3D(X_grid, Y_grid, cmap=cm.Blues, ax=None, file=None, show=False, label=None):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_xlabel('$x_1$', fontsize=13)
        ax.set_ylabel('$x_2$', fontsize=13)
        ax.set_zlabel('f(\mathbf{x})', fontsize=13)
        ax.plot_surface(X_grid, X_grid.T, Y_grid, cmap=cmap, label=label)
        if file is not None:
            plt.grid()
            plt.savefig(file, format='pdf')
        if show:
            plt.grid()
            plt.show()
        if file is None and show is False:
            return ax
        return None

    @staticmethod
    def parallel_coordinate(X, c='black', bounds=None, alpha=0.5, axs=None, file=None, show=False, ):
        n_var = X.shape[1]
        if axs is None:
            fig, axs = plt.subplots(1, n_var - 1, sharey=False)

        x = np.arange(1, n_var + 1)
        for i in range(n_var - 1):
            for j in range(X.shape[0]):
                axs[i].plot(x[0:n_var], X[j], ls='-', c=c, alpha=alpha)
            axs[i].set_xlim([x[i], x[i + 1]])
            axs[i].set_ylim(bounds[i])
            if i != n_var - 2:
                axs[i].set_xticks([x[i]])
                axs[i].set_xticklabels([str(x[i])])
            else:
                axs[i].set_xticks([x[i], x[i] + 1])
                axs[i].set_xticklabels([str(x[i]), str(x[i] + 1)])
            if i != 0:
                axs[i].set(yticklabels=[])
            axs[i].tick_params(axis='both', labelsize=13)
        plt.subplots_adjust(wspace=0)
        if file is not None:
            plt.grid()
            plt.savefig(file, format='pdf')
        if show:
            plt.grid()
            plt.show()
        if file is None and show is False:
            return axs
