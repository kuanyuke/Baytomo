#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:31:51 2021

@author: kuanyu
"""

import time
import logging
import numpy as np
import copy
import sys
from Baytomo import utils
logger = logging.getLogger()
rf_targets = ['prf', 'srf']
swd_targets = ['rdispph', 'ldispph', 'rdispgr', 'ldispgr']


class ObservedData(object):
    """
    The observed data object only consists of y.
    x = period or time (continuous and monotone increasing vector)
    y = y(x)
    stas = location of stations (x,y)
    pairs = source and receiver pairs based on the idx of stas (only used for RF)
        should be presented by idx of ([src , rev])
    """

    def __init__(self, x, y, stas, pairs=None, yerr=None, baz =None, bins=None):
        self.x = x
        self.y = y
        self.yerr = yerr

        self.stas = stas
        self.nstas = len(stas)
        self.pairs = pairs
        self.bazs = baz
        self.bins = bins

        if self.bins is None:
            self.bins = np.ones(self.nstas) 

        if self.yerr is None or np.any(yerr <= 0.) or np.any(np.isnan(yerr)):
            self.yerr = np.ones(self.y.shape) * np.nan

        if self.pairs is not None:
            # add or pair sort already!!! use np diff
            self.sortparis()
            self.nobs = len(self.x)
        else:
            self.nobs = len(self.y)
            self.pairsinprds = None

    def sortparis(self):
        """
        Sort pairs based on receiver indices.

        This method is useful for calculating ttm field for multiple receivers at the same time.
        """
        order = np.lexsort([self.pairs[:, 1], self.pairs[:, 0]])

        self.pairs = self.pairs[order]
        self.y = self.y[order]
        self.yerr = self.yerr[order]

        # Initialize pairsinprds dictionary
        # {2: [0, 1, 2], 3: [1]} key: prd, value: pairidx
        self.pairsinprds = {}
        self.nttms = 0
        for i, prd in enumerate(self.x):
            ttm = self.y[:, i]
            pairs = []
            for j in range(len(self.pairs)):
                if ttm[j] > 0:
                    self.nttms += 1
                    pairs.append(j)
            self.pairsinprds[prd] = pairs

        self.pair_dists = np.ones(len(self.pairs))
        # Populate pairsinprds with pairs having ttm at each period
        for i, pair in enumerate(self.pairs):
            xx1, yy1 = self.stas[pair[0]]
            xx2, yy2 = self.stas[pair[1]]
            dist = np.sqrt((xx1-xx2)**2+(yy1-yy2)**2)
            self.pair_dists[i] = dist

class ModeledData(object):
    """
    The modeled data object consists of x and y, which are initiated with nan,
    and will be computed during the inversion with the forward modeling tools.
    The plugins are python wrappers returning synthetic data, based on:
    RF: RFmini (Joachim Saul, GFZ, Posdam)
    SW: Surf96 (Rob Herrmann, St. Louis University, USA)
    You can easily update the plugin with your own code. Initiate the plugin
    with the necessary parameters and forward the instance to the
    update_plugin(instance) method. You can access this method through the
    SingleTarget object.
    The final method returning synthetic x and y data must be named
    self.run_model(h, vp, vs, rho, **kwargs). You can find a template with
    necessary plugin structure and method names in the defaults folder.
    Get inspired by the source code of the existing plugins.
    """

    def __init__(self, obsx, ref, stas, nobs, pairs=None, pairsinprds=None, bazs=None, n_jobs = 1):

        if ref in rf_targets:

            if bazs is None:
                from Baytomo.rfmini_modrf import RFminiModRF
                self.plugin = RFminiModRF(obsx, nobs, ref)
            else:
                from Baytomo.rfraysum import RFraysumModRF    
                self.plugin = RFraysumModRF(obsx, nobs, ref, bazs)

            self.xlabel = 'Time in s'

            self.obsx = obsx
            self.nobs = nobs
            self.ref = ref
            self.bazs = bazs

        elif ref in swd_targets:
            from Baytomo.surf96_modsw import surf_forward
            self.plugin = surf_forward(obsx, stas, pairs, pairsinprds, ref, n_jobs)
            self.xlabel = 'Period in s'

        else:
            message = "Please provide a forward modeling plugin for your " + \
                "target.\nUse target.update_plugin(MyForwardClass())"
            logger.info(message)
            self.plugin = None
            self.xlabel = 'x'

        self.x = np.nan
        self.y = np.nan

    def update(self, plugin):
        self.plugin = plugin


    def calc_synth(self,  vp, vs, ra, rho, von, **kwargs):
        """ Call forward modeling method of plugin."""
        self.x, self.y = self.plugin.run_model(
                vp=vp, vs=vs, ra=ra, rho=rho, von=von,**kwargs)
    
class Valuation(object):
    """
    Computation methods for likelihood and misfit are provided.
    The RMS misfit is only used for display in the terminal to get an estimate
    of the progress of the inversion.
    ONLY the likelihood is used for Bayesian inversion.
    """

    def __init__(self):
        self.corr_inv = None
        self.logcorr_det = None
        self.misfit = None
        self.likelihood = None

    @staticmethod
    def get_rms(yobs, ymod):
        """Return root mean square."""
        rms = np.sqrt(np.mean((ymod - yobs)**2))
        return rms

    @staticmethod
    def get_sigma_inv(corr, size):
        d = np.ones(size) + corr**2
        d[0] = d[-1] = 1
        e = np.ones(size-1) * -corr
        corr_inv = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)

    @staticmethod
    def get_covariance_nocorr(sigma, size, yerr=None, corr=0):
        """Return inverse and log-determinant of covariance matrix
        for a correlation (corr) of 0.
        If there is no correlation between data points, the correlation matrix
        is represented by the diagonal.
        """
        c_inv = np.diag(np.ones(size)) / (sigma**2)
        logc_det = (2*size) * np.log(sigma)
        return c_inv, logc_det

    @staticmethod
    def get_covariance_nocorr_scalederr(sigma, size, yerr, corr=0):
        """Return inverse and log-determinant of covariance matrix
        for a correlation (corr) of 0.
        If there is no correlation between data points, the correlation matrix
        is represented by the diagonal. Errors are relatively scaled.
        """
        scaled_err = yerr / yerr.min()

        c_inv = np.diag(np.ones(size)) / (scaled_err * sigma**2)
        logc_det = (2*size) * np.log(sigma) + np.log(np.product(scaled_err))
        return c_inv, logc_det

    @staticmethod
    def get_corr_inv(corr, size):
        d = np.ones(size) + corr**2
        d[0] = d[-1] = 1
        e = np.ones(size-1) * -corr
        corr_inv = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
        return corr_inv

    def get_covariance_exp(self, corr, sigma, size, yerr=None):
        """Return inverse and log-determinant of covariance matrix
        for a correlation (corr) not equaling 0.
        The correlation between data points is represented by an EXPONENTIAL law.
        """
        c_inv = self.get_corr_inv(corr, size) / (sigma**2 * (1-corr**2))
        logc_det = (2*size) * np.log(sigma) + (size-1) * np.log(1-corr**2)

        return c_inv, logc_det

    def init_covariance_gauss(self, corr, size, nobs, rcond=None):
        """
        Here fix the corr !!!
        """
        self.corr_inv = []
        self.logcorr_det = []
        idx = np.fromfunction(lambda i, j: (abs((i+j) - 2*i)),
                                  (size, size))                  
        rmatrix = corr[0]**(idx**2)
        if rcond is not None:
            corr_inv = np.linalg.pinv(rmatrix, rcond=rcond)
        else:
            corr_inv = np.linalg.inv(rmatrix)

        _, logdet = np.linalg.slogdet(rmatrix)
        logcorr_det = logdet

        for k in range(nobs):

            self.corr_inv.append(corr_inv)
            self.logcorr_det.append(logcorr_det)

    def get_covariance_gauss(self, sigma, size, idx, yerr=None, corr=None):
        """Return inverse and log-determinant of covariance matrix
        for a correlation (corr) not equaling 0.
        The correlation between data points is represented by a GAUSSIAN law.
        Consider this type of correlation if a gaussian filter was applied
        to compute RF. In this case, the inverse and log-determinant of the
        correlation matrix R is computed only once when initiating the chains.
        """

       
        c_inv = self.corr_inv[idx] / (sigma**2)
        logc_det = (2*size) * np.log(sigma) + self.logcorr_det[idx]
        return c_inv, logc_det

    @staticmethod
    def get_swdlikelihood(yobs, ymod, frac, sigma):
        """Return log-likelihood."""
        ydiff = ymod - yobs
        size = len(ydiff)

        madist = 0
        for i in range(size):
            madist_part = (ydiff[i]**2 / (sigma**2))
            madist += madist_part
        logL_part = size * np.log(sigma)
        logL = (logL_part - madist / 2.)
        return logL

    @staticmethod
    def get_rflikelihood(yobs, ymod, c_inv, logc_det):
        """Return log-likelihood."""
        ydiff = ymod - yobs
        madist = (ydiff.T).dot(c_inv).dot(ydiff)  # Mahalanobis distance
        logL_part = -0.5 * (yobs.size * np.log(2*np.pi) + logc_det)
        logL = logL_part - madist / 2.

        return logL


class SingleTarget(object):
    """A SingleTarget object gathers observed and modeled data,
    and the valuation methods. It provides methods to calculate misfit and
    likelihood, and also a plotting method. These can be used when initiating
    and testing your targets.
    """

    def __init__(self, x, y, ref, stas, pairs=None, pairsinprds=None, yerr=None, baz=None, bins = None, n_jobs = 1):
        self.ref = ref
        self.obsdata = ObservedData(
            x=x, y=y, stas=stas, pairs=pairs, yerr=yerr, baz=baz, bins = bins)
        self.moddata = ModeledData(obsx=x, ref=ref, stas=stas, nobs=self.obsdata.nobs,
                                   pairs=self.obsdata.pairs,
                                   pairsinprds=self.obsdata.pairsinprds,  bazs=self.obsdata.bazs, n_jobs = n_jobs)

        self.valuation = Valuation()

        logger.info("Initiated target: %s (ref: %s)"
                    % (self.__class__.__name__, self.ref))

    def update_plugin(self, plugin):
        self.moddata.update(plugin)

    def _moddata_valid(self):
        if self.moddata.y is np.nan or self.moddata.x is np.nan:
            return False
        if not type(self.moddata.x) == np.ndarray:
            return False
        if not len(self.obsdata.x) == len(self.moddata.x):
            return False
        if not np.sum(self.obsdata.x - self.moddata.x) <= 1e-5:
            return False
        if not (self.obsdata.y.size) == (self.moddata.y.size):
            return False

        return True


    def calc_misfit(self):
        if not self._moddata_valid():
            self.valuation.misfit = 1e15
            return

        misfits = 0
        sqaure_diff_sum = 0
        for n in range(self.obsdata.nobs):
            if self.noiseref == 'swd':
                obsdata = self.obsdata.y[:, n]
                moddata = self.moddata.y[:, n]
                sqaure_diff = sum((moddata - obsdata)**2)  # sum_station_pairs
                sqaure_diff_sum += sqaure_diff  # sum_freqs
            elif self.noiseref == 'rf':
                obsdata = self.obsdata.y[n]
                moddata = self.moddata.y[n]
                sqaure_diff = sum((moddata - obsdata)**2) 
                sqaure_diff_sum += sqaure_diff 
                #misfit = self.valuation.get_rms( obsdata, moddata)
                #misfits += misfit

                

        if self.noiseref == 'swd':       
            misfits = np.sqrt(sqaure_diff_sum/(self.obsdata.nttms))
        else:
            misfits = np.sqrt(sqaure_diff_sum/(self.obsdata.nobs))
       
        self.valuation.misfit = misfits


class RayleighDispersionPhase(SingleTarget):
    noiseref = 'swd'

    def __init__(self, x, y, stas, pairs, yerr=None,n_jobs = 1):
        ref = 'rdispph'
        SingleTarget.__init__(self, x, y, ref, stas, pairs, yerr=yerr, n_jobs =n_jobs)


class RayleighDispersionGroup(SingleTarget):
    noiseref = 'swd'

    def __init__(self, x, y, stas, pairs, yerr=None,n_jobs = 1):
        ref = 'rdispgr'
        SingleTarget.__init__(self, x, y, ref, stas, pairs, yerr=yerr, n_jobs =n_jobs)


class LoveDispersionPhase(SingleTarget):
    noiseref = 'swd'

    def __init__(self, x, y, stas, pairs, yerr=None,n_jobs = 1):
        ref = 'ldispph'
        SingleTarget.__init__(self, x, y, ref, stas, pairs, yerr=yerr, n_jobs =n_jobs)


class LoveDispersionGroup(SingleTarget):
    noiseref = 'swd'

    def __init__(self, x, y, stas, pairs, yerr=None,n_jobs = 1):
        ref = 'ldispgr'
        SingleTarget.__init__(self, x, y, ref, stas, pairs, yerr=yerr, n_jobs =n_jobs)


class PReceiverFunction(SingleTarget):
    noiseref = 'rf'

    def __init__(self, x, y, stas, baz=None, bins=None, yerr=None):
        ref = 'prf'
        SingleTarget.__init__(self, x, y, ref, stas, yerr=yerr, baz = baz, bins = bins)


class SReceiverFunction(SingleTarget):
    noiseref = 'rf'

    def __init__(self, x, y, stas, baz=0,yerr=None):
        ref = 'srf'
        SingleTarget.__init__(self, x, y, ref, stas, yerr=yerr, baz = baz, bins = bins)


class JointTarget(object):
    """
    A JointTarget object contains a list of SingleTargets and is responsible
    for computing the joint likelihood, given all model parameters.

    Attributes:
    - swdtargets (list): List of SingleTargets for seismic waveform data.
    - rftargets (list): List of SingleTargets for receiver function data.
    - nrftargets (int): Number of receiver function targets.
    - nswdtargets (int): Number of seismic waveform targets.
    - targets (list): Combined list of swdtargets and rftargets.
    - ntargets (int): Total number of targets.

    Methods:
    - get_ntargets(targets, ref): Get the total number of targets for a given reference ('rf' or 'swd').
    - get_misfits(): Compute misfit by summing target misfits.
    - calc_swdlikelihood(target, noise): Calculate the likelihood of seismic waveform data for a given target.
    - calc_rflikelihood(target, noise): Calculate the likelihood of receiver function data for a given target.
    - accept_as_currentmoddata: Accept the current model data as the proposed model data for all targets.
    - evaluate(von, vp, vs, ra, swdnoise, rfnoise, **kwargs): Evaluate the given model by computing the joint misfit and joint likelihoods for all targets.

    """

    def __init__(self, swdtargets=[], rftargets=[]):
        if len(rftargets) != 0:
            self.rftargets = rftargets
            self.nrftargets = self.get_ntargets(rftargets, ref='rf')
        else:
            self.nrftargets = 0

        if len(swdtargets) != 0:
            self.swdtargets = swdtargets
            self.nswdtargets = self.get_ntargets(swdtargets, ref='swd')
        else:
            self.nswdtargets = 0

        self.targets = swdtargets + rftargets  # list of SingleTargets
        self.ntargets = self.nswdtargets + self.nrftargets


        self.k=0

    def get_ntargets(self, targets, ref):
        """
        Get the total number of targets for a given reference ('rf' or 'swd').

        Parameters:
        - targets (list): List of SingleTargets for a specific reference.
        - ref (str): Reference type ('rf' or 'swd').

        Returns:
        - int: Total number of targets.

        """
        num = 0
        for n, target in enumerate(targets):
            nor = target.obsdata.nobs
            num = num + nor
        return num


    def get_misfits(self):
        """Compute misfit by summing target misfits.
        Keep targets' individual misfits for comparison purposes."""
        misfits = [target.valuation.misfit for target in self.targets]
        jointmisfit = np.sum(misfits)
        return np.concatenate((misfits, [jointmisfit]))


    def calc_swdlikelihood(self, target, noise):
        """
        Calculate the likelihood of seismic waveform data for a given target.

        Parameters:
        - target (object): The target object containing observational and model waveform data.
        - noise (array): An array containing noise parameters (fraction and sigma) for each period.

        Returns:
        - float: The log-likelihood of the seismic waveform data.

        """ 
        logL = 0
        prds = target.obsdata.x

        for n, prd in enumerate(prds):  # every period
            madist = 0
            frac, sigma = noise[2*n:2*n+2]
            

            ydiff = target.moddata.y[:, n] - target.obsdata.y[:, n]
            yerr = target.obsdata.yerr[:, n]

            # as in each prd containd different number of pair
            pairsidx = target.obsdata.pairsinprds[prd]
            size = len(pairsidx)

            
            for pair in pairsidx:
               madist_part = (ydiff[pair]**2 / (sigma**2))
               madist += madist_part



            logL_part = -0.5 * (np.log(2*np.pi)) - np.log(sigma)
            logL_target = (size * logL_part - madist /2.)

            logL += logL_target
        return logL

    def calc_rflikelihood(self, target, noise):
        """
        Calculate the likelihood of receiver function data for a given target.

        Parameters:
        - target (object): The target object containing observational and model receiver function data.
        - noise (array): An array containing noise parameters (correlation and sigma) for each receiver function.

        Returns:
        - float: The log-likelihood of the receiver function data.

        """

        logL = 0
        nor = target.obsdata.nobs

        diff_sum = 0
        for n in range(nor):
            ydiff = target.moddata.y[n] - target.obsdata.y[n]
            yerr = target.obsdata.yerr[n]
            diff_sum+=sum(ydiff)
            _, size = target.obsdata.y.shape  # time

            corr, sigma = noise[2*n:2*n+2]
            c_inv, logc_det = target.get_covariance(
                sigma=sigma, size=size, idx=n, yerr=yerr, corr=corr)
            
            madist = (ydiff.T).dot(c_inv).dot(ydiff)
            logL_part = -0.5 * (size * np.log(2*np.pi) + logc_det)

            if target.obsdata.bins[n] > 1:
                madist /= target.obsdata.bins[n]  # Adjust misfit term
       
            logL_target = logL_part - madist / 2.

            #logL_target = (logL_part - madist / ( (2.*target.obsdata.bins[n])))
            logL += logL_target

        return logL

    def accept_as_currentmoddata(self):
        """
        Accept the current model data as the proposed model data for all targets.

        """         
        for target in self.targets:
            if target.moddata.plugin.cacl:
                target.moddata.plugin.accept_as_currentmoddata()


    def evaluate(self, von, vp, vs, ra, swdnoise, rfnoise,  **kwargs):
        """
        Evaluate the given model by computing the joint misfit and joint likelihoods for all targets.
        This evaluation method basically evaluates the given model.
        It computes the jointmisfit, and more important the jointlikelihoods.
        The jointlikelihood (here called the proposallikelihood) is the sum
        of the log-likelihoods from each target.
        Parameters:
        - von (array): Von Mises stress data for the model.
        - vp (array): P-wave velocity data for the model.
        - vs (array): S-wave velocity data for the model.
        - ra (array): Radial anisotropy data for the model.
        - swdnoise (array): Noise parameters for seismic waveform data (fraction and sigma).
        - rfnoise (array): Noise parameters for receiver function data (correlation and sigma).
      
        - **kwargs: Additional keyword arguments.

        """

        # calc_synth :
        rho = kwargs.pop('rho', vp * 0.32 + 0.77)
        logL = 0
       
        self.proposallikelihoodjoint = np.zeros(len(self.targets))

        for n, target in enumerate(self.targets):

          
   
            target.moddata.calc_synth(
                        vp=vp, vs=vs, ra=ra, rho=rho, von=von,  **kwargs)
            
            if not target._moddata_valid():
                self.proposallikelihood = np.nan
                self.proposalmisfits = np.nan 

                return


            nor = target.obsdata.nobs
            target.calc_misfit()
            if target.noiseref == 'swd':
                noise, swdnoise = swdnoise[:2*nor], swdnoise[2*nor:]
                logL_target = self.calc_swdlikelihood(target, noise)
            elif target.noiseref == 'rf':
                noise, rfnoise = rfnoise[:2*nor], rfnoise[2*nor:]
                logL_target = self.calc_rflikelihood(target, noise)

            self.proposallikelihoodjoint[n] = logL_target 
            logL += logL_target


            
        
        self.proposallikelihood = logL
        self.proposalmisfits = self.get_misfits()
     
