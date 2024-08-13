#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:01:36 2022

@author: kuanyu
"""

import os
import glob
import logging
import numpy as np
import os.path as op
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import OrderedDict
import copy


from Baytomo import utils
from Baytomo import Targets
from Baytomo.Models import Model
from scipy.spatial import cKDTree as KDTree
import matplotlib.colors as colors
from joblib import Parallel, delayed, parallel_backend
rf_targets = ['prf', 'srf']
swd_targets = ['rdispph', 'ldispph', 'rdispgr', 'ldispgr']
logger = logging.getLogger(__name__)
rstate = np.random.RandomState(333)


def tryexcept(func):
    def wrapper_tryexcept(*args, **kwargs):
        try:
            output = func(*args, **kwargs)
            return output
        except Exception as e:
            print('* %s: Plotting was not possible\nErrorMessage: %s'
                  % (func.__name__, e))
            return None
    return wrapper_tryexcept


def vs_round(vs):
    # rounding down to next smaller 0.025 interval
    vs_floor = np.floor(vs)
    return np.round((vs-vs_floor)*40)/40 + vs_floor


class PlotFromStorage(object):
    """
    Plot and Save from storage (files).
    No chain object is necessary.
    """

    def __init__(self, configfile):
        """
        Initialize a PlotFromStorage object.

        Parameters:
        - configfile (str): Path to the configuration file.

        Attributes:
        - targets (list): List of target objects.
        - ntargets (int): Number of targets.
        - refs (list): List of target references.
        - nswdtargets (int): Number of seismic waveform targets.
        - nrftargets (int): Number of receiver function targets.
        - grids (dict): Dictionary containing grid information.
        - priors (dict): Dictionary containing prior information.
        - initparams (dict): Dictionary containing initial parameter information.
        - datapath (str): Path to the data directory.
        - figpath (str): Path to the figure directory.
        - mantle (dict): Mantle prior information.
        - refmodel (dict): Reference model information.

        """
        condict = self.read_config(configfile)

        self.targets = condict['targets']
        self.ntargets = len(self.targets)
        self.refs = condict['targetrefs'] + ['joint']
        self.nswdtargets = condict['nswdtargets']
        self.nrftargets = condict['nrftargets']

        self.grids = condict['grids']
        self.priors = condict['priors']
        self.initparams = condict['initparams']

        self.datapath = op.dirname(configfile)
        self.figpath = self.datapath.replace('data', '')

        self.init_filelists()
        self.init_outlierlist()
        self._sorted_chains()

        self.mantle = self.priors.get('mantle', None)

        self.refmodel = {'model': None,
                         'nlays': None,
                         'noise': None,
                         'vpvs': None}

    def read_config(self, configfile):
        """
        Read configuration from a file.

        Parameters:
        - configfile (str): Path to the configuration file.

        Returns:
        - dict: Configuration dictionary.

        """

        return utils.read_config(configfile)

    def savefig(self, fig, filename):
        """
        Save a figure to a file.

        Parameters:
        - fig: Matplotlib figure object.
        - filename (str): Name of the output file.

        """

        if fig is not None:
            outfile = op.join(self.figpath, filename)
            fig.savefig(outfile, bbox_inches="tight")
            plt.close('all')

    def init_outlierlist(self):
        """
        Initialize the list of outlier chains.

        """

        outlierfile = op.join(self.datapath, 'outliers.dat')
        if op.exists(outlierfile):
            self.outliers = np.loadtxt(outlierfile, usecols=[0], dtype=int)
            print('Outlier chains from file: %d' % self.outliers.size)
        else:
            print('Outlier chains from file: None')
            self.outliers = np.zeros(0)

    def init_filelists(self):
        """
        Initialize file lists for different types of files.

        """        
        filetypes = ['modelsvs', 'modelsra', 'modelsvon',
                     'likes', 'misfits', 'swdnoise', 'rfnoise', 'vpvs']
        filepattern = op.join(self.datapath, 'c???_p%d%s.txt')
        files = []
        size = []
        
        for ftype in filetypes:
            p1files = sorted(glob.glob(filepattern % (1, ftype)))
            p2files = sorted(glob.glob(filepattern % (2, ftype)))
            files.append([p1files, p2files])
            size.append(len(p1files) + len(p2files))
      
        if len(set(size)) == 1:
            self.vsfiles, self.rafiles, self.vonfiles, \
                self.likefiles, self.misfiles, self.swdnoisefiles, self.rfnoisefiles, self.vpvsfiles = files
            
        else:
            logger.info('You are missing files. Please check ' +
                        '"%s" for completeness.' % self.datapath)
            logger.info('(filetype, number): ' + str(zip(filetypes, size)))

    def _get_chaininfo(self):
        """
        Get information about chains, including chain indices and the number of models.

        Returns:
        - tuple: List of chain indices and list of number of models.

        """       
        nmodels = [len(np.loadtxt(file)) for file in self.likefiles[1]]
        chainlist = [self._return_c_p_t(file)[0] for file in self.likefiles[1]]
        return chainlist, nmodels

    def _return_c_p_t(self, filename, models=False):
        """
        Return chain index, phase number, and type of file from the filename.

        Parameters:
        - filename (str): Name of the file.
        - models (bool): If True, return only chain index and phase.

        Returns:
        - tuple: Chain index, phase number, and file type (if models=False).
                 Chain index and phase number (if models=True).

        """
        c, pt = op.basename(filename).split('.txt')[0].split('_')
        cidx = int(c[1:])

        if models:
            return cidx, pt
        else:
            phase, ftype = pt[:2], pt[2:]

            return cidx, phase, ftype

    def get_outliers(self, dev):
        """Detect outlier chains.
        The median likelihood from each chain (main phase) is computed.
        Relatively to the most converged chain, outliers are declared.
        Chains with a deviation of likelihood of dev % are declared outliers.
        Chose dev based on actual results.
        
        Parameters:
        - dev (float): Deviation threshold for declaring outliers.

        Returns:
        - array: Array of outlier chain indices.

        """
        nchains = len(self.likefiles[1])
        chainidxs = np.zeros(nchains) * np.nan
        chainmedians = np.zeros(nchains) * np.nan

        for i, likefile in enumerate(self.likefiles[1]):
            cidx, _, _ = self._return_c_p_t(likefile)
            chainlikes = np.loadtxt(likefile)
            chainmedian = np.median(chainlikes)

            chainidxs[i] = cidx
            chainmedians[i] = chainmedian

        maxlike = np.max(chainmedians)  # best chain average

        # scores must be smaller 1
        if maxlike > 0:
            scores = chainmedians / maxlike
        elif maxlike < 0:
            scores = maxlike / chainmedians

        outliers = chainidxs[np.where(((1-scores) > dev))]
        outscores = 1 - scores[np.where(((1-scores) > dev))]

        if len(outliers) > 0:
            print('Outlier chains found with following chainindices:\n')
            print(outliers)
            outlierfile = op.join(self.datapath, 'outliers.dat')
            with open(outlierfile, 'w') as f:
                f.write(
                    '# Outlier chainindices with %.3f deviation condition\n' % dev)
                for i, outlier in enumerate(outliers):
                    f.write('%d\t%.3f\n' % (outlier, outscores[i]))

        return outliers

    def get_outliers_misfit_joint(self, dev=1.3, threshold1 =False, threshold2 =False):
        """
        Detect outlier chains based on misfit values from different chains.
        Outliers are declared based on deviations from the most converged chain.

        Parameters:
        - dev (float): Deviation factor for declaring outliers.
        - threshold1 (float or bool): If provided, chains with median misfit above this value are declared outliers.
        - threshold2 (float or bool): If provided, chains with median misfit above this value are declared outliers.

        Returns:
        - outliers (np.array): Array of indices for outlier chains.
        """
        nchains = len(self.likefiles[0])
        chainidxs = np.zeros(nchains) * np.nan
        chainmedians1 = np.zeros(nchains) * np.nan
        chainmedians2 = np.zeros(nchains) * np.nan

        for i, misfile in enumerate(self.misfiles[0]):
            cidx, _, _ = self._return_c_p_t(misfile)
            chainmis1s = np.loadtxt(misfile).T[0]
            chainmis2s = np.loadtxt(misfile).T[1]
            chainmedian1 = np.median(chainmis1s[chainmis1s != 0]) 
            chainmedian2 = np.median(chainmis2s[chainmis2s != 0])

            chainidxs[i] = cidx
            chainmedians1[i] = chainmedian1
            chainmedians2[i] = chainmedian2

        minmis1 = np.min(chainmedians1)  # best chain average
        minmis2 = np.min(chainmedians2)  # best chain average
        scores1 = chainmedians1 / minmis1
        scores2 = chainmedians2 / minmis2


        outliers1 = chainidxs[np.where(((scores1) > dev))]
        outliers2 = chainidxs[np.where(((scores2) > dev))]
        outscores1 = 1 - scores1[np.where(((scores1) > dev))]
        outscores2 = 1 - scores2[np.where(((scores2) > dev))]
        outliers = np.concatenate((outliers1,outliers2 ))
        outscores = np.concatenate((outscores1,outscores2 ))

        if threshold1:
            outliers3 = chainidxs[chainmedians1 > threshold1]
            outscores3 = chainmedians1[chainmedians1 > threshold1]
            outliers4 = chainidxs[chainmedians2 > threshold2]
            outscores4 = chainmedians2[chainmedians2 > threshold2]

            # Combine with existing outliers
            combined_outliers = np.concatenate((combined_outliers, outliers3, outliers4))
            combined_scores = np.concatenate((combined_scores, outscores3, outscores4))

        # Remove duplicates: create a dictionary to ensure unique chainidxs
        outlier_dict = {}
        for idx, score in zip(combined_outliers, combined_scores):
            if idx not in outlier_dict or outlier_dict[idx] < score:
                outlier_dict[idx] = score

        # Convert dictionary back to arrays
        final_outliers = np.array(list(outlier_dict.keys()))
        final_outscores = np.array(list(outlier_dict.values()))

        if len(final_outliers) > 0:
            print('Outlier chains found with following chainindices:\n')
            outlierfile = op.join(self.datapath, 'outliers.dat')
            with open(outlierfile, 'w') as f:
                f.write(
                    '# Outlier chainindices with %.3f deviation condition\n' % dev)
                for i, outlier in enumerate(final_outliers):
                    f.write('%d\t%.3f\n' % (outlier, final_outscores[i]))

        return outliers

    def save_final_distribution(self, maxmodels=1000, dev=0.05,joint= False, threshold1=False,threshold2=False):
        """
        Save the final models from all chains, phase 2.
        As input, all the chain files in self.datapath are used.
        Outlier chains will be detected automatically using % dev. The outlier
        detection is based on the maximum reached (median) likelihood
        by the chains. The other chains are compared to the "best" chain and
        sorted out, if the likelihood deviates more than dev * 100 %.
        > Chose dev based on actual results.
        Maxmodels is the maximum number of models to be saved (.npy).
        The chainmodels are combined to one final distribution file,
        while all models are evenly thinned.
        """

        # def save_finalmodels(modelsvs, modelsra ,modelsvon, likes, misfits, swdnoise, rfnoise, vpvs):
        def save_finalmodels(vsmodels, ramodels, vonmodels, likes, misfits, swdnoise,  rfnoise, vpvs):
            """Save chainmodels as pkl file"""
            names = ['vsmodels', 'ramodels', 'vonmodels', 'likes',
                     'misfits', 'swdnoise', 'rfnoise', 'vpvs']
            for i, data in enumerate([vsmodels, ramodels, vonmodels, likes, misfits, swdnoise,  rfnoise, vpvs]):
                outfile = op.join(self.datapath, 'c_%s' % names[i])
                np.save(outfile, data)

        # delete old outlier file if evaluating outliers newly
        #outlierfile = op.join(self.datapath, 'outliers.dat')
        if op.exists(outlierfile):
            os.remove(outlierfile)

        if joint:
            self.outliers = self.get_outliers_misfit_joint(dev=dev,threshold1=threshold1,threshold2=threshold2)
        else:
            self.outliers = self.get_outliers(dev=dev)
        

        # due to the forced acceptance rate, each chain should have accepted
        # a similar amount of models. Therefore, a constant number of models
        # will be considered from each chain (excluding outlier chains), to
        # add up to a collection of maxmodels models.
        nchains = int(len(self.likefiles[0]))  - self.outliers.size
        maxmodels = int(maxmodels)
        mpc = int(maxmodels / nchains)  # models per chain
        # # open matrixes and vectors
        allmisfits = None
        allvsmodels = None
        allramodels = None
        allvonmodels = None
        alllikes = np.ones(maxmodels) * np.nan
        allswdnoise = np.ones((maxmodels, self.nswdtargets*2)) * np.nan
        allrfnoise = np.ones((maxmodels, self.nrftargets*2)) * np.nan
        allvpvs = np.ones(maxmodels) * np.nan
        ncellmax = self.priors['ncells'][1]

        start = 0
        chainidxs, nmodels = self._get_chaininfo()
        for i, cidx in enumerate(chainidxs):
            if cidx in self.outliers:
                continue
            index = np.arange(nmodels[i]).astype(int)
            if nmodels[i] > mpc:
                index = rstate.choice(index, mpc, replace=False)
                index.sort()

            chainfiles = [self.vsfiles[1][i], self.rafiles[1][i],self.vonfiles[1][i],
                          self.misfiles[1][i], self.likefiles[1][i], self.swdnoisefiles[1][i],
                          self.rfnoisefiles[1][i], self.vpvsfiles[1][i]]

            for c, chainfile in enumerate(chainfiles):
                _, _, ftype = self._return_c_p_t(chainfile)
                if ftype == 'modelsvon':
                    txtdata = np.genfromtxt(
                        chainfile, comments="#", delimiter='\n', dtype=None, encoding=None)
                    nucleus_x = np.array([np.fromstring(
                        v, dtype=float, sep=' ') for v in txtdata[::3]], dtype=object)[index]
                    nucleus_y = np.array([np.fromstring(
                        v, dtype=float, sep=' ') for v in txtdata[1::3]], dtype=object)[index]
                    nucleus_z = np.array([np.fromstring(
                        v, dtype=float, sep=' ') for v in txtdata[2::3]], dtype=object)[index]

                elif ftype == 'modelsvs' or ftype == 'modelsra':
                    txtdata = np.genfromtxt(
                        chainfile, delimiter='\n', dtype=None, encoding=None)
                    data = np.array([np.fromstring(v, dtype=float, sep=' ')
                                     for v in txtdata], dtype=object)
                    data = data[index]
               
                else:
                    data = np.loadtxt(chainfile)
                    data = data[index]
                    
                if c == 0:
                    end = start + len(data)

                if ftype == 'likes':
                    alllikes[start:end] = data

                elif ftype == 'modelsvs':
                    if allvsmodels is None:
                        allvsmodels = np.ones((maxmodels, ncellmax)) * np.nan
                    for i, vs in enumerate((data)):
                        allvsmodels[start:end, :len(vs)] = vs

                elif ftype == 'modelsra':
                    if allramodels is None:
                        allramodels = np.ones((maxmodels, ncellmax)) * np.nan
                    for i, ra in enumerate((data)):
                        allramodels[start:end, :len(ra)] = ra

                elif ftype == 'modelsvon':
                    if allvonmodels is None:
                        allvonmodels = np.ones(
                            (maxmodels, ncellmax, 3)) * np.nan
                    
                    for i, xcell in enumerate((nucleus_x)):
                        cell = np.stack(
                            (nucleus_x[i], nucleus_y[i], nucleus_z[i]), axis=1)
                        allvonmodels[start:end, :len(cell), :] = cell

                elif ftype == 'misfits':
                    if allmisfits is None:
                        allmisfits = np.ones(
                           (maxmodels, data[0].size)) * np.nan
                    allmisfits[start:end, :] = data

                elif ftype == 'swdnoise' and self.nswdtargets > 0:
                    allswdnoise[start:end, :] = data

                elif ftype == 'rfnoise' and self.nrftargets > 0:
                    allrfnoise[start:end, :] = data

                elif ftype == 'vpvs':
                    allvpvs[start:end] = data

            start = end

        # exclude nans
        allvsmodels = allvsmodels[~np.isnan(alllikes)]
        allramodels = allramodels[~np.isnan(alllikes)]
        allvonmodels = allvonmodels[~np.isnan(alllikes)]
        allmisfits = allmisfits[~np.isnan(alllikes)]
        allswdnoise = allswdnoise[~np.isnan(alllikes)]
        allrfnoise = allrfnoise[~np.isnan(alllikes)]
        allvpvs = allvpvs[~np.isnan(alllikes)]
        alllikes = alllikes[~np.isnan(alllikes)]
        save_finalmodels(allvsmodels, allramodels, allvonmodels,
                         alllikes, allmisfits, allswdnoise, allrfnoise, allvpvs)

    def _get_posterior_data(self, data, final, chainidx=0):
        """
        Retrieve posterior data arrays from files.

        Parameters:
        - data: List of dataset names to retrieve.
        - final: Boolean flag indicating if this is the final set of data.
        - chainidx: Index of the chain for non-final data.

        Returns:
        - List of numpy arrays containing the posterior data.
        """
        if final:
            filetempl = op.join(self.datapath, 'c_%s.npy')
        else:
            filetempl = op.join(
                self.datapath, 'c%.3d_p2%s.np?' % (chainidx, '%s'))

        outarrays = []
        for dataset in data:
            datafile = glob.glob(filetempl % dataset)[0]
            p2data = np.load(datafile)
            outarrays.append(p2data)
        return outarrays

    def _get_posterior_model(self, data, final, chainidx=0):
        """
        Retrieve posterior model arrays from files.

        Parameters:
        - data: List of dataset names to retrieve.
        - final: Boolean flag indicating if this is the final set of data.
        - chainidx: Index of the chain for non-final data.

        Returns:
        - List of numpy arrays containing the posterior models.
        """

        if final:
            filetempl = op.join(self.datapath, 'c_%smodels.npz')
        else:
            filetempl = op.join(
                self.datapath, 'c%.3d_p2%s.npy' % (chainidx, '%s'))

        outarrays = []
        for dataset in data:
            datafile = filetempl % dataset
            p2data = np.load(datafile)
            outarrays.append(p2data)
        return outarrays

    def _get_ncells(self, models):
        """
        Get the number of cells from models.

        Parameters:
        - models (list): List of arrays containing model data.

        Returns:
        - array: Array containing the number of cells for each model.

        """

        cellsnumber = np.array([(len(model))
                                for model in models])

        return cellsnumber

    def _sorted_chains(self, phase=0, ind=-1):
        """
        Sort chains based on their median misfit values for a given phase and index.

        Parameters:
        - phase: Integer index indicating which phase of the misfit files to use.
        - ind: Integer index indicating the column of misfit values to use.

        Updates:
        - self.sorted_chains: List of chain indices sorted by median misfit values.
        """
        # Retrieve chain information
        chainlist, _ = self._get_chaininfo()

        # Initialize arrays to store median misfits and corresponding chain indices
        num_chains = len(self.likefiles[0])
        chainmedians = np.full(num_chains, np.nan)
        chainidxs = np.full(num_chains, np.nan)

        # Loop through each misfit file to compute median misfit values
        for i, misfile in enumerate(self.misfiles[phase]):
            cidx, _, _ = self._return_c_p_t(misfile)
            misfit_values = np.loadtxt(misfile).T[ind]
            median_misfit = np.median(misfit_values)

            chainidxs[i] = cidx
            chainmedians[i] = median_misfit

        # Combine chain indices and median misfits, and sort by median misfit
        sorted_data = sorted(zip(chainidxs, chainmedians), key=lambda x: x[1])

        # Extract sorted chain indices
        self.sorted_chains = [item[0] for item in sorted_data]
       
    def _plot_iitervalues(self, files, ax, ncells=0, misfit=0, noise=0, ind=-1, reodercolor=False):
        """
        Plot values per iteration.

        Parameters:
        - files (list): List of file paths.
        - ax: Matplotlib axis.
        - ncells (bool): If True, plot the number of cells.
        - misfit (bool): If True, plot misfit values.
        - noise (bool): If True, plot noise values.
        - ind (int): Index for selecting specific values.

        Returns:
        - ax: Updated Matplotlib axis.

        """        
        unifiles = set([f.replace('p1', 'p2') for f in files])
        base = cm.get_cmap(name='rainbow')
        color_list = base(np.linspace(0, 1, len(unifiles)))

        xmin = -self.initparams['iter_burnin']
        xmax = self.initparams['iter_main']
        xmax = xmin + self.initparams['iter_burnin'] + self.initparams['iter_main']
        thinning = self.initparams['thinning']


        files.sort()
        n = 0
        for i, file in enumerate(files):
            phase = int(op.basename(file).split('_p')[1][0])
            alpha = (0.4 if phase == 1 else 0.7)
            ls = ('-' if phase == 1 else '-')
            lw = (0.5 if phase == 1 else 0.8)
            chainidx, _, _ = self._return_c_p_t(file)

            if reodercolor:
                idx = sorted_chains.index(chainidx)
                color = color_list[idx]
            else:
                color = color_list[n]

            if ncells:
                txtdata = np.genfromtxt(
                    file, delimiter='\n', dtype=None, encoding=None)
                data = np.array([np.fromstring(v, dtype=float, sep=' ')
                                 for v in txtdata], dtype=object)
                data = self._get_ncells(data)
            else:
                data = np.loadtxt(file)

            if misfit:
                data = data.T[ind]
            if noise:
                data = data.T[1::2][10]



            if phase == 1:

                xmed = xmin + data.size * thinning
                iters = (np.linspace(xmin, 0, data.size))
            else:
                xmed = 0 + data.size * thinning
                iters = (np.linspace(0, xmed, data.size))

            label = 'c%d' % (chainidx)
            ax.plot(iters, data, color=color,
                    ls=ls, lw=lw, alpha=alpha,
                    label=label if phase == 2 else '')

            if phase == 2:
                if n == 0:
                    datamax = data.max()
                    datamin = data.min()
                else:
                    datamax = np.max([datamax, data.max()])
                    datamin = np.min([datamin, data.min()])
                n += 1

        ax.set_xlim(xmin, xmax)
        if ncells:
            ax.set_ylim(0, int(datamax*1.2))
        elif misfit:
            ax.set_ylim(datamin*0.8, datamax*1.2)
        else:
            ax.set_ylim(datamin*0.8, datamax*1.2)

        ax.axvline(0, color='k', ls=':', alpha=0.7)

        (abs(xmin) + xmax)
        center = np.array([abs(xmin/2.), abs(xmin) + xmax/2.]
                          ) / (abs(xmin) + xmax)
        for i, text in enumerate(['Burn-in phase', 'Exploration phase']):
            ax.text(center[i], 0.97, text,
                    fontsize=12, color='k',
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=ax.transAxes)

        ax.set_xlabel('# Iteration')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return ax

    @tryexcept
    def plot_iitermisfits(self, nchains=6, ind=-1):
        """
        Plot misfit values per iteration.

        Parameters:
        - nchains (int): Number of chains to plot.
        - ind (int): Index for selecting specific values.

        Returns:
        - fig: Matplotlib figure.

        """
        files = self.misfiles[0][:nchains] + self.misfiles[1][:nchains]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax, misfit=True, ind=ind)
        ax.set_ylabel('%s misfit' % self.refs[ind])
        return fig

    @tryexcept
    def plot_iiterlikes(self, nchains=6):
        """
        Plot likelihood values per iteration.

        Parameters:
        - nchains (int): Number of chains to plot.

        Returns:
        - fig: Matplotlib figure.

        """
        files = self.likefiles[0][:nchains] + self.likefiles[1][:nchains]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax)
        ax.set_ylabel('Likelihood')
        # plt.show()
        return fig

    @tryexcept
    def plot_iiterswdnoise(self,  nchains=6, ind=-1):
        """
        Plot seismic waveform noise values per iteration.

        Parameters:
        - nchains (int): Number of chains to plot.
        - ind (int): Index for selecting specific period.

        nind = noiseindex, meaning:

        0: 'rfnoise_corr'  # should be const, if gauss
        1: 'rfnoise_sigma'
        2: 'swdnoise_corr'  # should be 0
        3: 'swdnoise_sigma'
        # dependent on number and order of targets.
        """
        
        
        for target in self.targets:
            files = self.swdnoisefiles[0][:nchains] + self.swdnoisefiles[1][:nchains]
            if target.noiseref == 'swd':
                nobs = target.obsdata.x
            if target.noiseref == 'rf':
                continue
        for obs in range(nobs):
            files = self.rfnoisefiles[0][:nchains] + self.rfnoisefiles[1][:nchains]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax = self._plot_iitervalues(files, ax, noise=True, ind=ind)

            parameter = np.concatenate(
                [['correlation (%s)' % ref, '$\sigma$ (%s)' % ref] for ref in self.refs[:-1]])
            ax.set_ylabel(parameter[ind])
        return fig

    @tryexcept
    def plot_iiterrfnoise(self,  nchains=6, ind=-1):
        """
        Plot receiver function noise values per iteration.

        Parameters:
        - nchains (int): Number of chains to plot.
        - ind (int): Index for selecting specific station.

        nind = noiseindex, meaning:
        0: 'rfnoise_corr'  # should be const, if gauss
        1: 'rfnoise_sigma'
        2: 'swdnoise_corr'  # should be 0
        3: 'swdnoise_sigma'
        # dependent on number and order of targets.
        """
        
        for target in self.targets:
            files = self.swdnoisefiles[0][:nchains] + self.swdnoisefiles[1][:nchains]
            if target.noiseref == 'swd':
                continue
            if target.noiseref == 'rf':
                nobs = len(target.obsdata.y)
        for obs in range(nobs):
            files = self.rfnoisefiles[0][:nchains] + self.rfnoisefiles[1][:nchains]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax = self._plot_iitervalues(files, ax, noise=True, ind=ind)

            parameter = np.concatenate(
                [['correlation (%s)' % ref, '$\sigma$ (%s)' % ref] for ref in self.refs[:-1]])
            ax.set_ylabel(parameter[ind])
        return fig

    @tryexcept
    def plot_iiterncells(self, nchains=6):
        """
        Plot the number of cells per iteration.

        Parameters:
        - nchains (int): Number of chains to plot.

        Returns:
        - fig: Matplotlib figure.

        """
        files = self.vsfiles[0][:nchains] + self.vsfiles[1][:nchains]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax, ncells=True)
        ax.set_ylabel('Number of cells')
        return fig

    @tryexcept
    def plot_iitervpvs(self, nchains=6):
        """
        Plot Vp/Vs values per iteration.

        Parameters:
        - nchains (int): Number of chains to plot.

        Returns:
        - fig: Matplotlib figure.

        """
        files = self.vpvsfiles[0][:nchains] + self.vpvsfiles[1][:nchains]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax)
        ax.set_ylabel('Vp / Vs')
        return fig


    def final_gridmodels(self, final=True, chainidx=0, n_jobs = 8, nsample =1000,refinenx=False):
        def run_kdtree(i,vs, ra, nucleus, vpvs):

            vs = vs[~np.isnan(vs)]
            ra = ra[~np.isnan(ra)]
            nucleus = nucleus[~np.isnan(nucleus).all(axis=-1)]
            if len(vs) != len(nucleus.T[0]):
              
                return None
     
            kdtree = KDTree(nucleus)

            kdtreedist, kdtreeidx = kdtree.query(self.gridsmodel)
            vp = vs * vpvs
            gridvp, gridvs, gridra = Model.kdtree_to_grid(vp=vp, vs=vs, ra=ra,
                                                          kdtreeidx=kdtreeidx)
            

            return (gridvp, gridvs, gridra)


        def get_layers(gridvp, gridvs,gridra,  nx, ny, nz):
                gridvp_xyplane = gridvp.reshape(nx*ny, nz)
                gridvs_xyplane = gridvs.reshape(nx*ny, nz)
                gridra_xyplane = gridra.reshape(nx*ny, nz)
                vel_dep_profile =[]
                for i in range(nx*ny):
                    	    
                    h, _, _, _, _ = Model.get_stepmodel_from_grids(
                        1, gridvp_xyplane[i], gridvs_xyplane[i], gridra_xyplane[i], gridvs_xyplane[i], 70)
                   
                    vel_dep_profile.append(np.cumsum(h))
                return  vel_dep_profile
   
        def save_final_gridmodels():
            names = ['meangridvp', 'meangridvs', 'meangridra',
                 'stdgridvp', 'stdgridvs', 'stdgridra']
            for i, data in enumerate([self.meangridvp, self.meangridvs, self.meangridra,
                                  self.stdgridvp, self.stdgridvs, self.stdgridra]):
                outfile = op.join(self.datapath, 'c_%s' % names[i])
                np.save(outfile, data)
            # Save the list using pickle
            import pickle
            outfile= op.join(self.datapath, 'c_deplayers.pkl')
            with open(outfile, 'wb') as f:
                pickle.dump(self.griddep, f)



        meangridvpfile = op.join(self.datapath, 'c_%s.npy' % 'meangridvp')
        if os.path.exists(meangridvpfile):
            self.meangridvp = np.load(meangridvpfile)
            self.meangridvs = np.load(
                meangridvpfile.replace('meangridvp', 'meangridvs'))
            self.meangridra = np.load(
                meangridvpfile.replace('meangridvp', 'meangridra'))
            self.stdgridvp = np.load(
                meangridvpfile.replace('meangridvp', 'stdgridvp'))
            self.stdgridvs = np.load(
                meangridvpfile.replace('meangridvp', 'stdgridvs'))
            self.stdgridra = np.load(
                meangridvpfile.replace('meangridvp', 'stdgridra'))
            return

        if final:
            vss, ras, vons, vpvss = self._get_posterior_data(
                ['vsmodels', 'ramodels', 'vonmodels', 'vpvs'], final=final)
        else:
            vss, ras,  vons, vpvss = self._get_posterior_data(
                ['modelsvs', 'modelsra', 'modelsvon', 'vpvs'], final, chainidx)
        
        nmodels = len(vss)

        allgridvp = []
        allgridvs = []
        allgridra = []
        allgriddep = []

        allstdgridvp = []
        allstdgridvs = []
        allstdgridra = []


        nsample = 1000
        nmodelsinjob = int(nmodels/nsample)
        if refinenx:
            grids = copy.deepcopy(self.grids)
            grids['nx'] = refinenx
            grids['ny'] = refinenx
            grids = Model.create_grid_model(copy.deepcopy(grids))
            self.gridsmodel = grids.gridsmodel
        else:
            grids = Model.create_grid_model(copy.deepcopy(self.grids))
            self.gridsmodel = grids.gridsmodel           


        for i in range(nmodelsinjob):

            start = int(i * nsample)
            end =  int((i+1)*nsample)
            vs = vss[start:end]
            ra = ras[start:end]
            von = vons[start:end]
            vpvs = vpvss[start:end]

            values = Parallel(n_jobs=n_jobs)(delayed(run_kdtree)(
            i, vs[i], ra[i], von[i], vpvs[i]) for i in range(len(vs)))

            gridvp = [item[0] for item in values  if item is not None]
            gridvs = [item[1] for item in values  if item is not None]
            gridra = [item[2] for item in values  if item is not None]

            meangridvp = np.array([np.mean(x) for x in zip(*gridvp)])
            meangridvs = np.array([np.mean(x) for x in zip(*gridvs)])
            meangridra = np.array([np.mean(x) for x in zip(*gridra)])
            stdgridvp = np.array([np.std(x) for x in zip(*gridvp)])
            stdgridvs = np.array([np.std(x) for x in zip(*gridvs)])
            stdgridra = np.array([np.std(x) for x in zip(*gridra)])
            vel_dep_profile = get_layers(meangridvp,meangridvs, meangridvs,grids['nx'], grids['ny'], grids['nz'])

            allgridvp.append(meangridvp)
            allgridvs.append(meangridvs)
            allgridra.append(meangridra)
            allgriddep.append( vel_dep_profile)

            allstdgridvp.append(stdgridvp)
            allstdgridvs.append(stdgridvs)
            allstdgridra.append(stdgridra)

        self.meangridvp = np.array([np.mean(x) for x in zip(*allgridvp)])
        self.meangridvs = np.array([np.mean(x) for x in zip(*allgridvs)])
        self.meangridra = np.array([np.mean(x) for x in zip(*allgridra)])

        self.stdgridvp = np.array([np.std(x) for x in zip(*allgridvp)])
        self.stdgridvs = np.array([np.std(x) for x in zip(*allgridvs)])
        self.stdgridra = np.array([np.std(x) for x in zip(*allgridra)])

        self.griddep = [np.concatenate((x)) for x in zip(*allgriddep)]

        
        save_final_gridmodels()


    def _plot_bestmodels(self, xslice=None, yslice=None, zslice=None):
        """
        Plot best models in 2D slices (horizontal, vertical) based on given slices.

        Parameters:
        - xslice: Optional float. X-coordinate for vertical slice.
        - yslice: Optional float. Y-coordinate for vertical slice.
        - zslice: Optional float. Z-coordinate for horizontal slice.

        Returns:
        - fig: The matplotlib figure object.
        - axes: Array of matplotlib Axes objects.
        """        
        nx, ny, nz = self.grids['nx'], self.grids['ny'], self.grids['nz']
        xmin, xmax = self.grids['gridx']
        ymin, ymax = self.grids['gridy']
        zmin, zmax = self.grids['gridz']
        
        vmin, vmax = self.priors['vs']

        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]},
                                 figsize=(5, 6.5))
        
        # Horizontal slice (z-axis)
        if zslice is not None:
            dz = (zmax - zmin) / (nz - 1) if nz > 1 else zmax - zmin
            indz = int((zslice - zmin) / dz)
            vs = self.meangridvs.reshape(nx * ny, nz)
            im = axes[0].imshow(vs[:, indz].reshape(nx, ny).T, vmin=vmin, vmax=vmax, extent=(xmin, xmax, ymin, ymax), cmap='seismic_r', aspect='auto')
            fig.colorbar(im, ax=axes[0], label='Vs in km')

            vs_std = self.stdgridvs.reshape(nx * ny, nz)
            im = axes[1].imshow(vs_std[:, indz].reshape(nx, ny).T, extent=(xmin, xmax, ymin, ymax), cmap='hot_r', aspect='auto')
            fig.colorbar(im, ax=axes[1], label='Standard deviation')

        # Vertical slice (x-axis)
        if xslice is not None:
            dx = (xmax-xmin)/(nx-1)
            indx = int((xslice-xmin)/dx)

            vs = self.meangridvs.reshape(nx, ny, nz)
            im = axes[0].imshow(vs[indx, :, :].T, vmin=vmin, vmax=vmax, origin="lower", extent=(
                ymin, ymax, zmin, zmax), cmap='seismic_r', aspect="auto") #jet_r #seismic_r
            clb = fig.colorbar(im, ax=axes[0])
            clb.ax.set_ylabel('Vs in km')
            axes[0].set_ylabel('depth in km')
            axes[0].invert_yaxis()
            

            vs_std = self.stdgridvs.reshape(nx, ny, nz)
            im = axes[1].imshow(vs_std[indx, :, :].T, origin="lower", extent=(
                ymin, ymax, zmin, zmax), cmap='hot_r', aspect="auto") #jet
            clb =fig.colorbar(im, ax=axes[1])
            clb.ax.set_ylabel('standard deviation')
            axes[1].set_ylabel('depth in km')
            axes[0].set_xlabel('Profile in km')
            
        
            axes[1].invert_yaxis()



        # Vertical slice (y-axis)
        if yslice is not None:
            dy = (ymax-ymin)/(ny-1)
            indx = int((yslice-ymin)/dy)
            vs = self.meangridvs.reshape(nx, ny, nz)
            im = axes[0].imshow(vs[:, indx, :].T, vmin=vmin, vmax=vmax, origin="lower", extent=(
                xmin, xmax, zmin, zmax), cmap='seismic_r', aspect="auto")
            clb = fig.colorbar(im, ax=axes[0])
            clb.ax.set_ylabel('Vs in km')
            axes[0].set_ylabel('depth in km')
            axes[0].invert_yaxis()


            vs_std = self.stdgridvs.reshape(nx, ny, nz)
            im = axes[1].imshow(vs_std[:, indx, :].T, origin="lower", extent=(
                xmin, xmax, zmin, zmax), cmap='hot_r', aspect="auto")
            clb = fig.colorbar(im, ax=axes[1])
            clb.ax.set_ylabel('standard deviation')
            axes[1].set_ylabel('depth in km')
            axes[1].set_xlabel('Profile in km')
            axes[1].invert_yaxis()

        return fig, axes

    @tryexcept
    def plot_posterior_models(self, final=True, chainidx=0, xslice=None, yslice=None, zslice=None, refinex=False):
        """
        Plot posterior models based on the given slices.

        Parameters:
        - final: Boolean flag indicating if final models should be plotted.
        - chainidx: Integer index of the chain to plot.
        - xslice: Optional float for x-slice plotting.
        - yslice: Optional float for y-slice plotting.
        - zslice: Optional float for z-slice plotting.
        - refine: Boolean flag to indicate if models should be refined.

        Returns:
        - fig: The matplotlib figure object.
        """
        if final:
            nchains = self.initparams['nchains'] - self.outliers.size
            self.final_gridmodels(final=final,refinex=refinex)
        else:
            nchains = 1
        files = []
        fig, ax = self._plot_bestmodels(xslice, yslice, zslice)
        return fig
