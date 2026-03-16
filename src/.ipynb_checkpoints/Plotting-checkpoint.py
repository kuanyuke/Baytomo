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
import matplotlib as mpl
from collections import OrderedDict
import copy
import pickle
from matplotlib.gridspec import GridSpec
import pandas as pd
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
        #self._sorted_chains()

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

    def _get_chaininfo(self, phase):
        """
        Get information about chains, including chain indices and the number of models.

        Returns:
        - tuple: List of chain indices and list of number of models.

        """       
        nmodels = [len(np.loadtxt(file)) for file in self.likefiles[phase]]
        chainlist = [self._return_c_p_t(file)[0] for file in self.likefiles[phase]]
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

    def get_outliers(self, dev, phase):
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
        nchains = len(self.likefiles[phase])
        chainidxs = np.zeros(nchains) * np.nan
        chainmedians = np.zeros(nchains) * np.nan

        for i, likefile in enumerate(self.likefiles[phase]):
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
        print ("get outliers here ", scores, outscores)
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

    def get_outliers_misfit(self, dev):
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

        for i, misfile in enumerate(self.misfiles[1]):
            cidx, _, _ = self._return_c_p_t(misfile)
            chainmisfits = np.loadtxt(misfile)
            chainmedian = np.median(chainmisfits)

            chainidxs[i] = cidx
            chainmedians[i] = chainmedian

        minmis = np.min(chainmedians)  # best chain average

        # scores must be smaller 1
        scores = chainmedians / minmis


        outliers = chainidxs[np.where(((scores) > dev))]
        outscores = scores[np.where(((scores) > dev))] -1 
        print (scores, outscores)
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


    def get_outliers_misfit_joint(self, dev=1.3, threshold1 =False, threshold2 =False, phase=1):
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
        nchains = len(self.likefiles[phase])
        chainidxs = np.zeros(nchains) * np.nan
        chainmedians1 = np.zeros(nchains) * np.nan
        chainmedians2 = np.zeros(nchains) * np.nan

        for i, misfile in enumerate(self.misfiles[phase]):
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

    def get_outliers_misfit_joint2(self, dev=1.3, thresholds=False, phase=1):
        """
        Detect outlier chains based on misfit values from different chains.
        Outliers are declared based on deviations from the most converged chain.

        Parameters:
        - dev (float): Deviation factor for declaring outliers.
        - thresholds (list of floats or bool): List of threshold values for each misfit chain.
        Chains with median misfit above the corresponding threshold are declared outliers.

        Returns:
        - outliers (np.array): Array of indices for outlier chains.
        """
        nchains = len(self.likefiles[0])
        chainidxs = np.zeros(nchains) * np.nan
        chainmedians = np.zeros(nchains) * np.nan # Adapt to the number of chains
        likelihood_outliers = []
        likelihood_outscores = []

        

        for i, likefile in enumerate(self.likefiles[phase]):
            cidx, _, _ = self._return_c_p_t(likefile)
            chainlikes = np.loadtxt(likefile)
            chainmedian = np.median(chainlikes)

            if np.isnan(chainmedian):
                likelihood_outliers.append(cidx)
                likelihood_outscores.append(np.nan)  # If cha
            else:
                chainidxs[i] = cidx
                chainmedians[i] = chainmedian
            
  

        maxlike = np.nanmax(chainmedians)  # Best chain average (maximizing likelihood)

        print('maxlike', maxlike)
        # Calculate scores for likelihood (ensuring they are less than 1)
        if maxlike > 0:
            scores = chainmedians / maxlike
        elif maxlike < 0:
            scores = maxlike / chainmedians
        print (scores)

        # Identify outliers based on deviation
        new_outliers = chainidxs[np.where(((1 - scores) > dev))]
        new_outscores = 1 - scores[np.where(((1 - scores) > dev))]

        # Append new outliers and scores to the existing lists
        likelihood_outliers.extend(new_outliers)
        likelihood_outscores.extend(new_outscores)


        # Handle misfit-based outlier detection if thresholds are provided
        misfit_outliers = []
        misfit_outscores = []

        if thresholds:
            misfit_chainmedians = np.zeros((nchains, len(self.misfiles[0]))) * np.nan

            for i, misfile in enumerate(self.misfiles[phase]):
                cidx, _, _ = self._return_c_p_t(misfile)
                chainmisfits = np.loadtxt(misfile).T
                for j in range(len(chainmisfits)):
                    misfit_chainmedians[i, j] = np.median(chainmisfits[j][chainmisfits[j] != 0])

            for j, threshold in enumerate(thresholds):
                    outliers_thr = chainidxs[misfit_chainmedians[:, j] > threshold]
                    outscores_thr = misfit_chainmedians[:, j][misfit_chainmedians[:, j] > threshold]
                    misfit_outliers.append(outliers_thr)
                    misfit_outscores.append(outscores_thr)

        # Combine likelihood and misfit outliers
        combined_outliers = np.concatenate((likelihood_outliers, *misfit_outliers))
        combined_scores = np.concatenate((likelihood_outscores, *misfit_outscores))

        # Remove duplicates and keep the worst scores
        outlier_dict = {}
        for idx, score in zip(combined_outliers, combined_scores):
            if idx not in outlier_dict or outlier_dict[idx] < score:
                outlier_dict[idx] = score

        # Convert dictionary back to arrays
        final_outliers = np.array(list(outlier_dict.keys()))
        final_outscores = np.array(list(outlier_dict.values()))

        if len(final_outliers) > 0:
            print('Outlier chains found with the following chain indices:\n')
            print(final_outliers)
            outlierfile = op.join(self.datapath, 'outliers.dat')
            with open(outlierfile, 'w') as f:
                f.write(
                    '# Outlier chain indices with %.3f deviation condition\n' % dev)
                for i, outlier in enumerate(final_outliers):
                    f.write('%d\t%.3f\n' % (outlier, final_outscores[i]))

        return final_outliers



    def save_final_distribution(self, maxmodels=1000, dev=0.05, thresholds=False, phase= 2):
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
        outlierfile = op.join(self.datapath, 'outliers.dat')
        if op.exists(outlierfile):
            os.remove(outlierfile)

        phase = phase -1 
        #self.outliers = self.get_outliers_misfit_joint(dev=dev,thresholds = thresholds,phase =phase)
        self.outliers = self.get_outliers(dev=dev,phase=phase)
        #self.outliers = self.get_outliers_misfit(dev=dev)
        self._sorted_chains(phase)

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
        chainidxs, nmodels = self._get_chaininfo(phase)
        for i, cidx in enumerate(chainidxs):
            if cidx in self.outliers:# or cidx == 10 or cidx == 22  or cidx == 27:
                continue
            index = np.arange(nmodels[i]).astype(int)
            if nmodels[i] > mpc:
                index = rstate.choice(index, mpc, replace=False)
                index.sort()

            print (cidx, len(index))
            chainfiles = [self.vsfiles[phase][i], self.rafiles[phase][i],self.vonfiles[phase][i],
                          self.misfiles[phase][i], self.likefiles[phase][i], self.swdnoisefiles[phase][i],
                          self.rfnoisefiles[phase][i], self.vpvsfiles[phase][i]]

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

    def _sorted_chains(self, phase=2, ind=-1):
        """
        Sort chains based on their median misfit values for a given index.

        Parameters:
        - ind: Integer index indicating the column of misfit values to use.

        Updates:
        - self.sorted_chains: List of chain indices sorted by median misfit values.
        """
        # Retrieve chain information
        chainlist, _ = self._get_chaininfo(phase)

        # Initialize arrays to store median misfits and corresponding chain indices
        num_chains = len(self.likefiles[0])
        chainmedians = np.full(num_chains, np.nan)
        chainidxs = np.full(num_chains, np.nan)

        # Loop through each misfit file to compute median misfit values
        phase = 1 if self.misfiles[1] and len(self.misfiles[1]) == len(self.misfiles[0]) else 0
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
       
    def _plot_iitervalues(self, files, ax, ncells=0, misfit=0, noise=0, ind=-1, noiseind=0,reodercolor=False):
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
        unifiles = set([f.replace('p2', 'p1') for f in files])
        unifiles = set(files)
        base = cm.get_cmap(name='rainbow')
        color_list = base(np.linspace(0, 1, 30))#  len(unifiles)))# len(unifiles)))
        
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
  
            idx = self.sorted_chains.index(chainidx)
            color = color_list[idx]
 

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
                data = data.T[1::2][noiseind]

            
            if phase == 1:

                xmed = xmin + data.size * thinning
                iters = (np.linspace(xmin, xmed, data.size))
            else:
                xmed = 0 + data.size * thinning
                iters = (np.linspace(0, xmed, data.size))

            label = 'c%d' % (chainidx)
            ax.plot(iters, data, color=color,
                    ls=ls, lw=lw, alpha=alpha,
                    label=label if phase == 1 else '')

            if phase == 1:
                if n == 0:
                    if misfit:
                        datamax = data[-1000:].max()
                        datamin = data[-1000:].min()                            
                    else:
                        datamax = data.max()
                        datamin = data.min()
                else:
                    if misfit:
                        datamax = np.max([datamax, data[-1000:].max()])
                        datamin = np.min([datamin, data[-1000:].min()])                          
                    else:
                        datamax = np.max([datamax, data.max()])
                        datamin = np.min([datamin, data.min()])
                n += 1

        ax.set_xlim(xmin, xmax)
        if ncells:
            
            ax.set_ylim(0, int(datamax*1.2))
        elif misfit:
            ax.set_ylim(datamin*0.8, datamin*2)
            ax.set_ylim(0.0,5)#datamin*2)
            #ax.set_ylim(0.0, 0.1)#datamin*2)
        elif noise:
            ax.set_ylim(0.001, 0.012)
            ax.set_ylim(0.005, 2.0)
            ax.set_ylim(0., 0.05)
        else:
            ax.set_ylim(datamax*0.8, datamax*1.2)
            #ax.set_ylim(datamin*0.8, 0.012)
            #ax.set_ylim(datamin*0.8, 0.5)
            #ax.set_ylim(datamax*0.5, datamax*1.2)
        
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
        return fig

    @tryexcept
    def plot_iiterswdnoise(self,  nchains=6, ind=-1, noiseind=0):
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

        #for obs in range(nobs):
        files = self.swdnoisefiles[0][:nchains] + self.swdnoisefiles[1][:nchains]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax, noise=True, noiseind=noiseind)

        parameter = np.concatenate(
                [['correlation (%s)' % ref, '$\sigma$ (%s)' % ref] for ref in self.refs[:-1]])
        
        ax.set_ylabel(parameter[ind])
        return fig

    @tryexcept
    def plot_iiterrfnoise(self,  nchains=6, ind=-1, noiseind=0):
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
            files = self.rfnoisefiles[0][:nchains] + self.rfnoisefiles[1][:nchains]
            if target.noiseref == 'swd':
                continue
            if target.noiseref == 'rf':
                nobs = len(target.obsdata.y)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax = self._plot_iitervalues(files, ax, noise=True, noiseind=noiseind)

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

    def _plot_posterior_distribution(self, data, bins, formatter='%.2f', ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(3.5, 3))

        count, bins, patches = ax.hist(data, bins=bins, color='darkblue', alpha=0.7,
                                 edgecolor='white', linewidth=0.4)

        
        cbins = (bins[:-1] + bins[1:]) / 2.
        mode = cbins[np.argmax(count)]
        median = np.median(data)

        if formatter is not None:
            text = 'median: %s' % formatter % median
            ax.text(0.97, 0.97, text,
                    fontsize=9, color='k',
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes)

        ax.axvline(median, color='k', ls=':', lw=1)
        
        ax.set_yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
            
        return ax

    @tryexcept
    def plot_posterior_misfits(self, final=True, chainidx=0):
        misfits, = self._get_posterior_data(['misfits'], final, chainidx)
        figures, axes = plt.subplots(1,len(self.targets)+1,  figsize=(21,7))
        
        bins = 20
        formatter = '%.2f'
        for i, ax in enumerate(axes):
            ax = self._plot_posterior_distribution( misfits[:,i], bins, formatter, ax=ax)
            if i < len(self.targets):
                ax.set_xlabel('%s'%self.targets[i].ref)
            else:
                ax.set_xlabel('joint')
        return ax.figure

    @tryexcept
    def plot_posterior_ncells(self, final=True, chainidx=0,ax=None):

        models, = self._get_posterior_data(['vsmodels'], final, chainidx)
        
        # get ncells
        models = [model[~np.isnan(model)] for model in models]
        ncells = np.array([(len(model)) for model in models])

        bins = np.arange(np.min(ncells), np.max(ncells)+2)

        formatter = '%d'
        ax = self._plot_posterior_distribution( ncells, bins, formatter)


        ax.set_xlabel('Number of ncells')
        return ax.figure

    @tryexcept
    def plot_posterior_vpvs(self, final=True, chainidx=0):
        vpvs, = self._get_posterior_data(['vpvs'], final, chainidx)
        bins = 20
        formatter = '%.2f'

        ax = self._plot_posterior_distribution( vpvs, bins, formatter)
        ax.set_xlabel('$V_P$ / $V_S$')
        return ax.figure

    def get_noise(self,final=True, chainidx=0):
        noises = []
        
        for target in self.targets:
            if target.noiseref == 'rf':
                rftarget= target
                noise, = self._get_posterior_data(['rfnoise'], final, chainidx)
                
            else:
                swdtarget = target
                noise, =  self._get_posterior_data(['swdnoise'], final, chainidx)
            noises.append(noise)
        return noises

    @tryexcept
    def plot_posterior_noise(self, stanames=None,final=True, chainidx=0):
        
        noises = self.get_noise()
       

        figures = []
        axes = []
        
        for t, target in enumerate(self.targets):
            if target.noiseref == 'rf':
                nobs = len(target.obsdata.y)            
            elif target.noiseref ==  'swd':
                nobs = len(target.obsdata.x)
            noise = noises[t]
            nfigs = int(nobs/40)+1
            idx= 0
            bins = 20
            formatter = '%.4f'
            for f in range(nfigs): 
                fig, ax = plt.subplots(8,5, figsize=(14,20))
            

                fig.tight_layout()
                fig.subplots_adjust(hspace=0.32)    


  
                for j in range(8):
                    for i in range(5):
                        if idx >= nobs:
                        # Remove the unused subplot
                            fig.delaxes(ax[j][i])
                            continue
                        if target.noiseref ==  'swd':
                            ax[j][i].set_title("prd %s s" % target.obsdata.x[idx])
                
                        elif target.noiseref == 'rf':
                            ax[j][i].set_title('station of %s'% (stanames[idx]))

                               
                    
                        data = noise.T[1::2][idx]
                    
                        self._plot_posterior_distribution( data, bins, formatter, ax=ax[j][i])
                        ax[j][i].patch.set_facecolor('none')
                        ax[j][i].patch.set_alpha(0.5) 

                        idx += 1
                        
                if  target.noiseref ==  'swd':
                    fig.text(0.5, -0.01, '$\sigma_{swd}$ in km/s', va='center', ha='center', fontsize=mpl.rcParams['axes.labelsize'])    
                elif target.noiseref == 'rf':
                    fig.text(0.5, -0.01, '$\sigma_{rf}$', va='center', ha='center', fontsize=mpl.rcParams['axes.labelsize'])                
                figures.append(fig)
                axes.append(ax)
            


        return figures

    def _unique_legend( self, handles, labels):
        # if a key is double, the last handle in the row is returned to the key
        legend = OrderedDict(zip(labels, handles))
        return legend.values(), legend.keys()

    def haversine_distance(lat1, lon1, lat2, lon2, radius=6371e3):
        """
        Calculate the great-circle distance between two points on a sphere using the haversine formula.
        Args:
            lat1, lon1: Latitude and longitude of the first point (in degrees).
            lat2, lon2: Latitude and longitude of the second point (in degrees).
            radius: Radius of the sphere (default is Earth's radius in meters).
        Returns:
            Great-circle distance in meters.
        """
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = radius * c
        return distance


    def plot_vel(self, target, mod=True):
        nobs = len(target.obsdata.y)
        nfigs = int(nobs/12) + 1
        
        figlists = []
        figures = []
        axes = []
        nplot_obs = 0
        pair_dist = []

        if target.noiseref == 'rf':
            ylabel = 'Amplitude'
        elif target.noiseref == 'swd':
            ylabel = 'Travel time in s'
        
        for i in range(nfigs):
            fig, ax = plt.subplots(3,4, figsize=(14,7))
            fig.tight_layout()
            for j in range(12):
                x = int(j/4)
                y = int(j%4)
                nplot_obs = int(nplot_obs)
                
                if nplot_obs >= nobs:
                    fig.delaxes(ax[x][y])
                else:
                    if target.noiseref == 'swd':

                        y1 = target.obsdata.y[nplot_obs]
                        ttm = y1[np.where(y1>0)]
                        prd = target.obsdata.x[np.where(y1>0)]
                        yerr = target.obsdata.yerr[nplot_obs][np.where(y1>0)]
                        pair =  target.obsdata.pairs[nplot_obs]
                        xx1, yy1 = target.obsdata.stas[pair[0]]
                        xx2, yy2 = target.obsdata.stas[pair[1]]
                        if greatcircle:
                            dist = haversine_distance(xx1, yy1,xx2, yy2 )
                        else:
                            dist = np.sqrt((xx1-xx2)**2+(yy1-yy2)**2)
                        ttm = (np.ones(len(ttm))*dist)/ttm
                        pair_dist.append(dist)
                        ax[x][y].set_ylim(2.85, 4)
                    else:
                        ttm = target.obsdata.y[nplot_obs]
                        prd = target.obsdata.x                
                        yerr = target.obsdata.yerr[nplot_obs]#[np.where(y1>0)]
                    ax[x][y].errorbar(prd, ttm, yerr=yerr,
                        label='obs', marker='x', ms=1, color='b', lw=0.8,
                        elinewidth=0.7, zorder=1000)


                nplot_obs += 1

            fig.text(0.5, 0.00, target.moddata.xlabel, va='center', ha='center')
            fig.text(0.00, 0.5,  ylabel, va='center', ha='center', rotation='vertical')
            fig.text(0.5, 1,  target.ref, va='center', ha='center')

            filname = 'c_bestdatafits_%s%s'%(target.ref, i)
            figlists.append(filname)
            figures.append(fig)
            axes.append(ax)


        return figlists, figures, axes,pair_dist
    import copy

    def plot_obsveldata( self, ax=None, mod=False):
        """Return subplot of all targets."""
        figlists= []
        figures=[]
        axes = []
        pair_dists = []
        for i, target in enumerate(self.targets):
            if  target.noiseref == 'swd':
                figlist, fig, ax, pair_dist = self.plot_vel(target)
            else:
                figlist, fig, ax, _ = self.plot_vel(target)
                pair_dist=0
            figlists.append(figlist)
            figures.append(fig)
            axes.append(ax)
            pair_dists.append(pair_dist)

        return figlists, figures, axes, pair_dists

    def load_txt(self, file):
            txtdata = np.genfromtxt(file,delimiter='\n',dtype=None,encoding=None)
            data = np.array([np.fromstring(v, dtype=float, sep=' ') for v in txtdata],dtype=object) 
            return data
    def load_vontxt(self,file):
            txtdata = np.genfromtxt(file, comments="#",delimiter='\n',dtype=None,encoding=None)
            nucleus_x = np.array([np.fromstring(v, dtype=float, sep=' ') for v in txtdata[::3]],dtype=object)
            nucleus_y = np.array([np.fromstring(v, dtype=float, sep=' ') for v in txtdata[1::3]],dtype=object)
            nucleus_z = np.array([np.fromstring(v, dtype=float, sep=' ') for v in txtdata[2::3]],dtype=object)
            

            return [nucleus_x,nucleus_y,nucleus_z]

    def plot_datafits(self,stafile ):
        """Plot best data fits from each chain and ever best,
        ignoring outliers."""
        
        swdtarget= []
        rftarget=[]
        for target in (self.targets):
            if target.noiseref == 'swd':
                swdtarget.append(target)
            else:
                rftarget.append(target)
            
        
        targets = Targets.JointTarget(swdtargets = swdtarget, rftargets=rftarget)
        
        
        figlists, figs,axes, pair_dists = self.plot_obsveldata()
        pair_dist = pair_dists[0]

        thebestmodel = np.nan
        thebestmisfit = 1e15
        thebestchain = np.nan
        

        vsfiles = self.vsfiles[1]
        
        unifiles = vsfiles
        base = cm.get_cmap(name='rainbow')
        color_list = base(np.linspace(0, 1, len(unifiles)))

        df = pd.read_csv(stafile, usecols=[0,1])
        stanames = df.set_index('srcidx').T.to_dict('list') 
        
        for i, vsfile in enumerate(vsfiles):
            chainidx, _, _ = self._return_c_p_t(vsfile)
            if chainidx in self.outliers:
                continue
            idx = self.sorted_chains.index(chainidx)
            color = color_list[idx]



            vsmodels = self.load_txt(vsfile)
            ramodels = self.load_txt(vsfile.replace('modelsvs', 'modelsra'))
            vonmodels = self.load_vontxt(vsfile.replace('modelsvs', 'modelsvon'))
            
            vpvs = np.loadtxt(vsfile.replace('modelsvs', 'vpvs'))
            misfits = np.loadtxt(vsfile.replace('modelsvs', 'misfits')).T[-1]
            rfnoises = np.loadtxt(vsfile.replace('modelsvs', 'rfnoise')).T[1::2].T
            
            
            bestvs = vsmodels[np.argmin(misfits)]
            bestra = ramodels[np.argmin(misfits)]
            bestvon = [vonmodels[0][np.argmin(misfits)],vonmodels[1][np.argmin(misfits)],vonmodels[2][np.argmin(misfits)]]
            bestvpvs = vpvs[np.argmin(misfits)]
            bestmisfit = misfits[np.argmin(misfits)]
            bestrfnoise = rfnoises[np.argmin(misfits)]
         
            
            if bestmisfit < thebestmisfit:
                thebestmisfit = bestmisfit
                thebestvs = bestvs
                thebestra = bestra
                thebestvon = bestvon
                thebestvpvs = bestvpvs
                thebestchain = chainidx
            
            nucleus = np.stack((bestvon[0],bestvon[1], bestvon[2]),axis=1)
            kdtree = KDTree(nucleus)
        
            kdtreedist, kdtreeidx = kdtree.query(self.grids.gridsmodel)
            
            bestvp = bestvs * bestvpvs
            gridvon, gridvp, gridvs, gridra = Model.kdtree_to_grid(nucleus=nucleus,vp=bestvp, vs=bestvs, ra=bestra,
                                                          kdtreeidx=kdtreeidx)
            rho = gridvp * 0.32 + 0.77

            for n, target in enumerate(targets.targets):
                if target.noiseref == 'rf':
                    xmod, ymod = target.moddata.plugin.run_model( gridvp, gridvs, gridra, rho)#,  gridvon)#,  rfinit=False)
               
                    
                else:
                    xmod, ymod = target.moddata.plugin.run_model( gridvp, gridvs, gridra, rho)
               

                nfigs = int(len(target.obsdata.y)/12)+1
                
                for k in range (len(target.obsdata.y)):
                    i = int(k/12)
                    j = k-12*i
                    x = int(j/4)
                    y = int(j%4)
                    if target.noiseref == 'swd':
                        
                        y1 = target.obsdata.y[k]
                        ttm = ymod[k][np.where(y1>0)]
                        vel = np.ones(len(ttm))*pair_dist[k]/ttm
                        xmod = target.obsdata.x[np.where(y1>0)]
                        pair =  target.obsdata.pairs[k]
                        axes[n][i][x][y].set_title('Pair of %s and %s'% (pair[0], pair[1]))
                        axes[n][i][x][y].plot(xmod, vel, color=color, alpha=0.5, lw=1,label='mod')
                    else:
                        if chainidx ==0:
                            axes[n][i][x][y].plot([0,30],[0,0], '--', color = 'grey')   
                            axes[n][i][x][y].plot([0,0],[-0.2, 0.2], '--', color = 'grey')   
                            axes[n][i][x][y].set_xlim(-5,30)
                            axes[n][i][x][y].set_ylim(-0.12,0.18)
                            axes[n][i][x][y].text(20, 0.1 , '%s'% (stanames[k][0]), fontsize=8, fontweight = 'bold', zorder=100)
                        
                        rf =  ymod[k]
                        xmod = target.obsdata.x
                        axes[n][i][x][y].plot(xmod, rf, color=color, alpha=0.5, lw=1,label='mod', zorder=10)

                        
                        
                    axes[n][i][x][y].legend().set_visible(False)
                
                for i in range(nfigs):
                    han, lab = axes[n][i][0][0].get_legend_handles_labels()
                    handles, labels = self._unique_legend(han, lab)
                    #axes.legend().set_visible(False)
                    figs[n][i].legend(handles, labels, loc='center left', bbox_to_anchor=(0.92, 0.5))
                
                    outfile = op.join(self.figpath, figlists[n][i])
                    figs[n][i].savefig(outfile, bbox_inches="tight")

        return figs



    def final_gridmodels(self, final=True, chainidx=0, n_jobs = 4, nsample =1000,refinenx=False):
        def run_kdtree(i,vs, ra, nucleus, vpvs):

            vs = vs[~np.isnan(vs)]
            ra = ra[~np.isnan(ra)]
            nucleus = nucleus[~np.isnan(nucleus).all(axis=-1)]
            if len(vs) != len(nucleus.T[0]):    
                return None
     
            kdtree = KDTree(nucleus)

            kdtreedist, kdtreeidx = kdtree.query(self.gridsmodel)
            vp = vs * vpvs
            gridvon, gridvp, gridvs, gridra = Model.kdtree_to_grid(nucleus=nucleus,vp=vp, vs=vs, ra=ra,
                                                      kdtreeidx=kdtreeidx)

            return (gridvp, gridvs, gridra)


        def get_layers(gridvp, gridvs,gridra,  nx, ny, nz,zmax):
                gridvp_xyplane = gridvp.reshape(nx*ny, nz)
                gridvs_xyplane = gridvs.reshape(nx*ny, nz)
                gridra_xyplane = gridra.reshape(nx*ny, nz)
                vel_dep_profile =[]
                for i in range(nx*ny):
                    	    
                    h, _, _, _, _ = Model.get_stepmodel_from_grids(
                        1, gridvp_xyplane[i], gridvs_xyplane[i], gridra_xyplane[i], gridvs_xyplane[i], zmax)
                   
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



        #meangridvpfile = op.join(self.datapath, 'c_%s.npy' % 'meangridvp')
        #if os.path.exists(meangridvpfile):
        #    self.meangridvp = np.load(meangridvpfile)
        #    self.meangridvs = np.load(
        #        meangridvpfile.replace('meangridvp', 'meangridvs'))
        #    self.meangridra = np.load(
        #        meangridvpfile.replace('meangridvp', 'meangridra'))
        #    self.stdgridvp = np.load(
        #        meangridvpfile.replace('meangridvp', 'stdgridvp'))
        #    self.stdgridvs = np.load(
        #        meangridvpfile.replace('meangridvp', 'stdgridvs'))
        #    self.stdgridra = np.load(
        #        meangridvpfile.replace('meangridvp', 'stdgridra'))
        #    return

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

        nmodelsinjob = int(nmodels/nsample)
        if nmodels%nsample != 0:
            nmodelsinjob += 1

 
        grids = Model.create_grid_model(copy.deepcopy(self.grids))
        self.gridsmodel = grids.gridsmodel           
        print ("vss shape", len(vss))
        for i in range(nmodelsinjob):
            start = int(i * nsample)
            #if i == nmodelsinjob-1:
            #    end =  start + nmodels%nsample
            #else:
            end =  int((i+1)*nsample)

            vs = vss[start:end]
            ra = ras[start:end]
            von = vons[start:end]
            vpvs = vpvss[start:end]
            print ("Vs shape", i, vs.shape, start,end)

            with parallel_backend('loky', inner_max_num_threads=2):
                values = Parallel(n_jobs=n_jobs, timeout=300)(delayed(run_kdtree)(
                i, vs[i], ra[i], von[i], vpvs[i]) for i in range(len(vs)))

            gridvp = [item[0] for item in values  if item is not None]
            gridvs = [item[1] for item in values  if item is not None]
            gridra = [item[2] for item in values  if item is not None]

            del values

            meangridvp = np.array([np.mean(x) for x in zip(*gridvp)])
            meangridvs = np.array([np.mean(x) for x in zip(*gridvs)])
            meangridra = np.array([np.mean(x) for x in zip(*gridra)])
            stdgridvp = np.array([np.std(x) for x in zip(*gridvp)])
            stdgridvs = np.array([np.std(x) for x in zip(*gridvs)])
            stdgridra = np.array([np.std(x) for x in zip(*gridra)])

            zmin, zmax = self.grids['gridz']
            print ("grid shape", i, meangridvs.shape, meangridvp.shape)
            vel_dep_profile = get_layers(meangridvp,meangridvs, meangridvs,grids['nx'], grids['ny'], grids['nz'],zmax)

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




    def final_gridmodels_ai(self, final=True, chainidx=0, n_jobs = 8, nsample =1000,refinenx=False):
        def get_layers(gridvp, gridvs, gridra, nx, ny, nz, zmax):
            gridvp_xyplane = gridvp.reshape(nx*ny, nz)
            gridvs_xyplane = gridvs.reshape(nx*ny, nz)
            gridra_xyplane = gridra.reshape(nx*ny, nz)
            vel_dep_profile = []
            for i in range(nx*ny):
                h, _, _, _, _ = Model.get_stepmodel_from_grids(
                    1, gridvp_xyplane[i], gridvs_xyplane[i], gridra_xyplane[i], gridvs_xyplane[i], zmax)
                vel_dep_profile.append(np.cumsum(h))
            return vel_dep_profile


        def run_kdtree(i, vs, ra, nucleus, vpvs, grids, zmax):
            vs = vs[~np.isnan(vs)]
            ra = ra[~np.isnan(ra)]
            nucleus = nucleus[~np.isnan(nucleus).all(axis=-1)]
            if len(vs) != len(nucleus.T[0]):    
                return None
            kdtree = KDTree(nucleus)
            kdtreedist, kdtreeidx = kdtree.query(self.gridsmodel)
            vp = vs * vpvs
            gridvon, gridvp, gridvs, gridra = Model.kdtree_to_grid(nucleus=nucleus, vp=vp, vs=vs, ra=ra, kdtreeidx=kdtreeidx)
            vel_dep_profile = get_layers(gridvp, gridvs, gridra, grids['nx'], grids['ny'], grids['nz'],zmax)
            return (gridvp, gridvs, gridra,vel_dep_profile )

   
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


        if final:
                vss, ras, vons, vpvss = self._get_posterior_data(['vsmodels', 'ramodels', 'vonmodels', 'vpvs'], final=final)
        else:
                vss, ras, vons, vpvss = self._get_posterior_data(['modelsvs', 'modelsra', 'modelsvon', 'vpvs'], final, chainidx)
        
        nmodels = len(vss)
        grids = Model.create_grid_model(copy.deepcopy(self.grids))
        self.gridsmodel = grids.gridsmodel
        zmin, zmax = self.grids['gridz']

        # Initialize accumulators
        sum_vp = np.zeros(self.gridsmodel.shape[0], dtype=np.float64)
        sum_vs = np.zeros_like(sum_vp)
        sum_ra = np.zeros_like(sum_vp)
        sum_sq_vp = np.zeros_like(sum_vp)
        sum_sq_vs = np.zeros_like(sum_vp)
        sum_sq_ra = np.zeros_like(sum_vp)
        allgriddep = []
        nmodels_valid = 0

    
        nmodelsinjob = (nmodels + nsample - 1) // nsample  # Ceiling division
    
        for i in range(nmodelsinjob):
            start = i * nsample
            end = min((i+1)*nsample, nmodels)
            vs_batch = vss[start:end]
            ra_batch = ras[start:end]
            von_batch = vons[start:end]
            vpvs_batch = vpvss[start:end]
    
            values = Parallel(n_jobs=n_jobs)(
            delayed(run_kdtree)(idx, vs, ra, nucleus, vpvs, grids, zmax)
                    for idx, (vs, ra, nucleus, vpvs) in enumerate(zip(vs_batch, ra_batch, von_batch, vpvs_batch))
                    )

            valid_values = [v for v in values if v is not None]
            current_batch_size = len(valid_values)
            if current_batch_size == 0:
                continue


            gridvp_batch = [v[0] for v in valid_values]
            gridvs_batch = [v[1] for v in valid_values]
            gridra_batch = [v[2] for v in valid_values]
            layers_batch = [v[3] for v in valid_values]

            # Stack batch arrays and accumulate
            sum_vp += np.sum(np.stack(gridvp_batch, axis=0), axis=0)
            sum_vs += np.sum(np.stack(gridvs_batch, axis=0), axis=0)
            sum_ra += np.sum(np.stack(gridra_batch, axis=0), axis=0)
        
            sum_sq_vp += np.sum(np.stack(gridvp_batch, axis=0)**2, axis=0)
            sum_sq_vs += np.sum(np.stack(gridvs_batch, axis=0)**2, axis=0)
            sum_sq_ra += np.sum(np.stack(gridra_batch, axis=0)**2, axis=0)

            allgriddep.append( layers_batch)

    
            nmodels_valid += current_batch_size
    
        if nmodels_valid == 0:
            raise ValueError("No valid models processed.")
    
        # Compute global mean and std
        self.meangridvp = sum_vp / nmodels_valid
        self.meangridvs = sum_vs / nmodels_valid
        self.meangridra = sum_ra / nmodels_valid
    
        self.stdgridvp = np.sqrt((sum_sq_vp / nmodels_valid) - (self.meangridvp ** 2))
        self.stdgridvs = np.sqrt((sum_sq_vs / nmodels_valid) - (self.meangridvs ** 2))
        self.stdgridra = np.sqrt((sum_sq_ra / nmodels_valid) - (self.meangridra ** 2))
    
        # Compute depth profile for the final mean model
        zmin, zmax = self.grids['gridz']
        nx, ny, nz = grids['nx'], grids['ny'], grids['nz']
        #self.griddep = get_layers(self.meangridvp, self.meangridvs, self.meangridra, nx, ny, nz, zmax)
        self.griddep = [np.concatenate((x)) for x in zip(*allgriddep)]

        save_final_gridmodels()



    def plot_profile(self, x0, y0, x1, y1,  plot_hist= True):
        nx, ny, nz = self.grids['nx'], self.grids['ny'], self.grids['nz']
        xmin, xmax = self.grids['gridx']
        ymin, ymax = self.grids['gridy']
        zmin, zmax = self.grids['gridz']
        dx = (xmax-xmin)/(nx-1)
        dy=dx
    
        vmin, vmax = self.priors['vs']

        meangridvpfile = op.join(self.datapath, 'c_%s.npy' % 'meangridvp')
        meangridvp = np.load(meangridvpfile)
        meangridvs = np.load(meangridvpfile.replace('meangridvp', 'meangridvs'))
        meangridra = np.load( meangridvpfile.replace('meangridvp', 'meangridra'))

        stdgridvp = np.load(meangridvpfile.replace('meangridvp', 'stdgridvp'))
        stdgridvs = np.load(meangridvpfile.replace('meangridvp', 'stdgridvs'))
        stdgridra = np.load(meangridvpfile.replace('meangridvp', 'stdgridra'))
        with open(meangridvpfile.replace('meangridvp.npy', 'deplayers.pkl'), 'rb') as f:
            allgriddep = pickle.load(f)

        vs = meangridvs.reshape(nx, ny, nz)
        stdvs =stdgridvs.reshape(nx, ny, nz)
        def get_line(x0, y0, x1, y1):
                points = []
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                sx = 1 if x0 < x1 else -1
                sy = 1 if y0 < y1 else -1
                err = dx - dy

                while True:
                    points.append((x0, y0))
                    if x0 == x1 and y0 == y1:
                        break
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x0 += sx
                    if e2 < dx:
                        err += dx
                        y0 += sy
                return points
        
        # Convert coordinates to grid indices
        point1_x = int((x0 - xmin) / dx)
        point1_y = int((y0 - ymin) / dy)
        point2_x = int((x1 - xmin) / dx)
        point2_y = int((y1 - ymin) / dy)

        distance = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)

        # Get line coordinates
        line_coords = get_line(point1_x, point1_y, point2_x, point2_y)
    
        data = allgriddep
        # Make sure to fill data with the correct number of arrays for demonstration
        data = data * (nx*ny // len(data))  # Ensure we have nx*ny arrays

        # Convert the list to a 2D list structure
        reshaped_data = [data[i * nz:(i + 1) * nz] for i in range(nz)]

        # Convert the 2D list to a numpy object array
        allgriddep = np.array(reshaped_data, dtype=object)

        # Extract vs values at line coordinates
        profile_vs = []
        profile_stdvs = []
        profile_dep = []
        profile_dep_hist= []
        for x, y in line_coords:
            profile_vs.append(vs[x, y,:])
            profile_stdvs.append(stdvs[x, y,:])
            #profile_dep_hist.append(allgriddep[x,y])

            #heights, bins = np.histogram(allgriddep[x, y], bins=np.arange(0, nz,1))
            #profile_dep.append(heights)
        profile_vs = np.array(profile_vs)

        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]}, figsize=(10, 6.5))


        im = axes[0].imshow(np.array(profile_vs).T, vmin=vmin, vmax=vmax, cmap='jet_r', aspect="auto", extent=(
                0, distance, zmax, 0))
        clb = fig.colorbar(im, ax=axes[0])
        clb.ax.set_ylabel('Vs in km')
        axes[0].set_ylabel('depth in km')

        im = axes[1].imshow(np.array(profile_stdvs).T,  cmap='hot_r', aspect="auto", extent=(
                0, distance, zmax, 0))
        clb = fig.colorbar(im, ax=axes[1])
        clb.ax.set_ylabel('standard deviation')

        axes[0].set_ylim(zmax,zmin)
        axes[1].set_ylim(zmax,zmin)

        axes[1].set_ylabel('depth in km')
        axes[0].set_xlabel('Profile in km')

        width = 300  # km
        depth_range = (0, 150)  # Depth range for the uppermost layer (Moho to bottom)

        x_values = np.linspace(0, width, 301)


        # Calculate the depth of the Moho at each point along the x-axis with varying dip
        dip_range= (5,45)
        dip_range= (30,30)
    
        moho_depth = np.interp(x_values, [0, 100, 200, 300], [28, 28, depth_range[0], depth_range[1]])  # Initial values

        # Adjust the depth for the increasing dip segment
    
        for i in range(100, 201):
            moho_depth[i] = moho_depth[99] + (x_values[i] - x_values[99]) * np.tan(np.radians(dip_range[0]))

        for i in range(201, 301):
            moho_depth[i] = moho_depth[200] + (x_values[i] - x_values[200]) * np.tan(np.radians(dip_range[1]))

        axes[0].plot(x_values,moho_depth,  '--', color = 'grey')
        axes[1].plot(x_values,moho_depth, '--', color = 'grey')




        if plot_hist:
            fig2, axes = plt.subplots(1, len(profile_dep_hist), figsize=(10, 3.5), sharey=True)
            
            for i, hist_data in enumerate(profile_dep_hist):
            
                axes[i].hist(hist_data, bins=(zmax-zmin), orientation='horizontal',
                     color='lightgray', alpha=0.7,
                     edgecolor='k')
           
                axes[i].invert_yaxis() 
                axes[i].set_ylim(zmax,zmin) 
                axes[i].set_xticks([])
                plt.subplots_adjust(wspace=0)
            axes[0].set_ylabel('depth in km')
            return fig, fig2
        return fig


    @tryexcept
    def plot_posterior_models(self, final=True, chainidx=0, xslice=None, yslice=None, zslice=None, refinenx=False):
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
            self.final_gridmodels(final=final,refinenx=refinenx)
        else:
            nchains = 1
        files = []
        fig, ax = self._plot_bestmodels(xslice, yslice, zslice)
        return fig
