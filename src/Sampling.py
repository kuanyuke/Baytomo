#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:01:37 2023

@author: kuanyu
"""
import numpy as np
import os.path as op
import os
from Baytomo import utils
from scipy.spatial import cKDTree as KDTree
import scipy.spatial.distance
import copy
import glob

import logging
logger = logging.getLogger()

PAR_MAP = {'vsmod': 0, 'ramod': 1, 'vmod': 2, 'birth': 3, 'death': 3,
           'swdnoise': 5, 'rfnoise': 6, 'vpvs': 7, "vmod2":8}


PROPDIST = {'vsmod': 0, 'ramod': 1, 'vmod': 2, 'vsbirth/death': 3, 'rabirth/death': 4,
            'swdnoise': 5, 'rfnoise': 6, 'vpvs': 7,  "vmod2":8}


class Sampling(object):
    def __init__(self,  chainidx,  priors, grids, propdist, swd=False, swdnoisepriors=None, swdnoiseinds=None, rf=False, ramod=False, rstate=None):
        self.chainidx = chainidx
        self.priors = priors
        self.grids = grids
        self.propdist = propdist
        
        self.swd = swd
        self.swdnoisepriors = swdnoisepriors
        self.swdnoiseinds =swdnoiseinds # this term can be removed!
        self.ramod=ramod
        
        self.rf = rf
        self.rstate = rstate
        
        self.dvs2 = 0
        self.dra2=0
        self.ind_death=None
        self.birthnucleus=None

    def draw_initvpvs(self, ncells):
        if type(self.priors['vpvs']) == np.float64:
            return self.priors['vpvs']

        vpvsmin, vpvsmax = self.priors['vpvs']

        return self.rstate.uniform(low=vpvsmin, high=vpvsmax)
    


    def draw_initmodel(self, grids):
        # prior paremeters
        keys = self.priors.keys()
        vsmin, vsmax = self.priors['vs']
        self.ncellmin, self.ncellmax = self.priors['ncells']



        ncells = self.ncellmin

        # randomly choose vs, ra and assign to every  nucleu
        vsmean = 0.5 * (vsmin + vsmax)
        vs = self.rstate.normal(vsmean, 0.2, size=ncells)
       # vs  = self.rstate.uniform(low=vsmin, high=vsmax, size=ncells)


        vs.sort()

        nucleus_x = np.ones(ncells)*(0.5*(grids.xmin+grids.xmax))
        nucleus_y = np.ones(ncells)*(0.5*(grids.ymin+grids.ymax))
        nucleus_z = self.rstate.uniform(
            grids.zmin, grids.zmax, size=ncells)
        nucleus_z.sort()
        nucleus = np.vstack((nucleus_x, nucleus_y, nucleus_z)).T
 

        if self.ramod:
            ramin, ramax = self.priors['ra']
            ra = self.rstate.uniform(low=ramin, high=ramax, size=ncells)
        else:
            ra = np.ones(ncells) * self.priors['ra']

        model = [vs, ra, nucleus]
        return (model if self._validmodel(model)
               else self.draw_initmodel(grids))

    def draw_init_swdnoiseparams0(self, targets):
        if not self.swd:
            return [], []
        # for each target the noiseparams are (fraction and sigma)
        noiserefs = ['noise_frac', 'noise_sigma']
        init_noise = np.ones(targets.nswdtargets*2) * np.nan
        fracfix = np.zeros(targets.nswdtargets*2, dtype=bool)
        self.swdnoisepriors = []
        init_idx = 0
        
        for target in targets.swdtargets:
            for i in range(target.obsdata.nobs):
                for j, noiseref in enumerate(noiserefs):
                    idx = (2*i)+j + init_idx
                    noiseprior = self.priors[target.noiseref + noiseref]

                    if type(noiseprior) in [int, float, np.float64]:
                        fracfix[idx] = True
                        init_noise[idx] = noiseprior
                    else:
                        init_noise[idx] = self.rstate.uniform(
                            low=noiseprior[0], high=noiseprior[1])

                    self.swdnoisepriors.append(noiseprior)
            init_idx = 2 * target.obsdata.nobs

        self.swdnoiseinds = np.where(fracfix == 0)[0]
        return init_noise, fracfix

    def draw_init_rfnoiseparams0(self, targets):#
        if not self.rf:
            return [], []


        noiserefs = ['noise_corr', 'noise_sigma']
        init_noise = np.ones(targets.nrftargets*2) * np.nan
        corrfix = np.zeros(targets.nrftargets*2, dtype=bool)
        

        self.rfnoisepriors = []
        for target in targets.rftargets:
            for i in range(target.obsdata.nobs):
                for j, noiseref in enumerate(noiserefs):
                    idx = (2*i)+j
                    noiseprior = self.priors[target.noiseref + noiseref]
                    if type(noiseprior) in [int, float, np.float64]:
                        corrfix[idx] = True
                        init_noise[idx] = noiseprior
                    else:
                        init_noise[idx] = self.rstate.uniform(
                            low=noiseprior[0], high=noiseprior[1])

                    self.rfnoisepriors.append(noiseprior)
        self.rfnoiseinds = np.where(corrfix == 0)[0]
        if len(self.rfnoiseinds) == 0:
            logger.warning('All your noise parameters are fixed. On Purpose?')
        return init_noise, corrfix

    def draw_init_swdnoiseparams(self, targets):
        if not self.swd:
            return [], []
        # for each target the noiseparams are (fraction and sigma)
        noiserefs = ['noise_frac', 'noise_sigma']
        init_noise = np.ones(targets.nswdtargets*2) * np.nan
        fracfix = np.zeros(targets.nswdtargets*2, dtype=bool)
        self.swdnoisepriors = []
        init_idx = 0
        
        for target in targets.swdtargets:
            for j, noiseref in enumerate(noiserefs):
                num_low_high = int(0.25 * target.obsdata.nobs)
                noiseprior = self.priors[target.noiseref + noiseref]
                if type(noiseprior) in [int, float, np.float64]:
                    init_fracfix = True
                    init_noise_value = noiseprior

                else:
                    init_fracfix = False
                    init_noise_value = self.rstate.uniform(
                            low=noiseprior[0], high=noiseprior[1])
                    
                    if   targets.nrftargets !=0 and noiseprior[1] > 0.5:
                        noisemax = 0.5
                    else:
                        noisemax = noiseprior[1] 
                    init_noise_value =  self.rstate.uniform(noiseprior[0], noisemax) 


                self.swdnoisepriors.append(noiseprior)            
                for i in range(target.obsdata.nobs):
                    idx = (2*i)+j + init_idx
                    fracfix[idx] = init_fracfix
                    init_noise[idx] = init_noise_value

            init_idx = 2 * target.obsdata.nobs

        self.swdnoiseinds = np.where(fracfix == 0)[0]
        self.swdcounters = np.ones(len(self.swdnoiseinds))  # Initialize counters for swdnoiseinds

        return init_noise, fracfix

    def draw_init_rfnoiseparams(self, targets):#, rfinit):
        # for each target the noiseparams are (corr and sigma)
        if not self.rf:
            return [], []


        noiserefs = ['noise_corr', 'noise_sigma']
        init_noise = np.ones(targets.nrftargets*2) * np.nan
        corrfix = np.zeros(targets.nrftargets*2, dtype=bool)



        self.rfnoisepriors = []
        for target in targets.rftargets:
            for j, noiseref in enumerate(noiserefs):
                noiseprior = self.priors[target.noiseref + noiseref]
                if type(noiseprior) in [int, float, np.float64]:
                    init_corrfix = True
                    init_noise_value = noiseprior
                else:
                    init_corrfix = False
                    init_noise_value = self.rstate.uniform(
                            low=noiseprior[0], high=noiseprior[1])
                    #if  targets.nswdtargets !=0 and noiseprior[1] < 0.01:
                    if  targets.nswdtargets !=0 and noiseprior[0] < 0.005:
                        noisemin= 0.005
                    #if  targets.nswdtargets !=0 and noiseprior[1] >= 0.01:
                    ##    noisemin= 0.005
                     #   noisemax = 0.01
                    else:
                        noisemin = noiseprior[0] 
                        #noisemax = noiseprior[1] 
                    
                    init_noise_value =  self.rstate.uniform(noisemin, noiseprior[1]) 
                    #init_noise_value =  self.rstate.uniform(noiseprior[0] , noisemax) 
                    #init_noise_value = 0.005

                self.rfnoisepriors.append(noiseprior)
                for i in range(target.obsdata.nobs):
                    idx = (2*i)+j
                    corrfix[idx] = init_corrfix
                    init_noise[idx] = init_noise_value

                 
       
        self.rfnoiseinds = np.where(corrfix == 0)[0]
        self.rfcounters = np.ones(len(self.rfnoiseinds))  # Initialize counters for rfnoiseinds
        
       
        if len(self.rfnoiseinds) == 0:
            logger.warning('All your noise parameters are fixed. On Purpose?')
        return init_noise, corrfix



    def set_rftarget_covariance(self, targets, corrfix, inoise, rcond=None):
        "here decide which rf correlation should be used, need more check"
        # RF noise hyper-parameters: if corr is not 0, but fixed, the
        # correlation between data points will be ass[0]umed gaussian (realistic).
        # if the prior for RFcorr is a range, the computation switches
        # to exponential correlated noise for RF, as gaussian noise computation
        # is too time expensive because of computation of inverse and
        # determinant each time _corr is perturbed

        if not self.rf:
            return

        nnobs = 0
    
        for i, target in enumerate(targets.rftargets):
            nobs = target.obsdata.nobs
            
            start = nnobs
            end = nnobs + nobs
            target_corrfix = corrfix[::2][start:end]
            target_noise_corr = inoise[::2][start:end]
            
            if not target_corrfix[0]:
                # if False, means if corr is not fixed, exponentail
                # exponential for each target
                target.get_covariance = target.valuation.get_covariance_exp
                continue

            # gauss for RF
            if target.noiseref == 'rf':     
                size = target.obsdata.x.size
                target.valuation.init_covariance_gauss(
                    target_noise_corr, size, nobs, rcond=rcond)
                target.get_covariance = target.valuation.get_covariance_gauss

            else:
                message = 'The noise correlation automatically defaults to the \
exponential law. Explicitly state a noise reference for your user target \
(target.noiseref) if wished differently.'
                logger.info(message)
                # target.noiseref == 'swd'
                # target.get_covariance = target.valuation.get_covariance_nocorr
            nnobs = end
        #return targets

    def init_lastmodel_fromchain(self, datapath=None, chainidx=0):
        """
        Initialize the model parameters from the last available saved data in the specified chain.

        Parameters:
            - datapath (str): The path to the directory containing saved model data.
            - chainidx (int): The index of the chain from which to load the last available data.

        Returns:
            - Tuple: A tuple containing the initialized model parameters and noise values.
            - List: [ivs, ira, ivon] - Initial values for Vs, Ra, and VON parameters.
            - ndarray: iswdnoise - Initial values for SWD noise parameters.
            - ndarray: irfnoise - Initial values for RF noise parameters.
            - ndarray: ivpvs - Initial values for VP/VS ratio.
            - int: iiter - The iteration index of the last available data.

        """

        filetypes = ['modelsvs', 'modelsra', 'modelsvon', 'swdnoise', 'rfnoise', 'vpvs']
        
        filepattern = op.join(datapath, 'c%.3d_p%d%s.txt')
        files = []
        size = []

        # First: decide phase (check using modelsvs only)
        pattern_phase1 = filepattern % (chainidx, 2, 'modelsvs')
        
        if glob.glob(pattern_phase1):
            phase = 2
        else:
            phase = 1


        index = -1
        for i, ftype in enumerate(filetypes):

            chainfile = sorted(glob.glob(filepattern % (chainidx, phase,  ftype)))[0]


            if ftype == 'modelsvon':
                txtdata = np.genfromtxt(
                    chainfile, comments="#", delimiter='\n', dtype=None, encoding=None)
                nucleus_x = np.array([np.fromstring(v, dtype=float, sep=' ')
                                     for v in txtdata[::3]], dtype=object)#[1:]
                nucleus_x = nucleus_x[index]
                nucleus_y = np.array([np.fromstring(v, dtype=float, sep=' ')
                                     for v in txtdata[1::3]], dtype=object)#[1:]
                nucleus_y = nucleus_y[index]
                nucleus_z = np.array([np.fromstring(v, dtype=float, sep=' ')
                                     for v in txtdata[2::3]], dtype=object)#[1:]
                nucleus_z = nucleus_z[index]
                data = np.stack((nucleus_x, nucleus_y, nucleus_z), axis=1)
            elif ftype == 'modelsvs' or ftype == 'modelsra':
                txtdata = np.genfromtxt(
                    chainfile, delimiter='\n', dtype=None, encoding=None)
                data = np.array([np.fromstring(v, dtype=float, sep=' ')
                                for v in txtdata], dtype=object)
                if i == 0:
                    iiter = len(data)
                data = data[index]

            else:
                data = np.loadtxt(chainfile)
                data = data[index]
            files.append(data)



        ivs, ira, ivon, iswdnoise, irfnoise, ivpvs = files

        return [ivs, ira, ivon], iswdnoise, irfnoise, ivpvs, iiter

    def update_prodist(self, propdist):
        self.propdist = propdist
        
    def sampling(self,modify, model, vpvs, rfnoise, swdnoise, currentkdtree ):
        self.currentmodel = model
        self.currentrfnoise = None
        self.currentvpvs = vpvs
        self.currentrfnoise = rfnoise
        self.currentswdnoise = swdnoise
        self.currentkdtree = currentkdtree
        self.vnoi_move2 = False

        if modify in ['vsmod', 'ramod', 'vmod', 'zvmod', 'zbirth', 'birth', 'death']:
            proposalmodel = self._get_modelproposal(modify)
            proposalrfnoise = self.currentrfnoise
            proposalswdnoise = self.currentswdnoise
            proposalvpvs = self.currentvpvs
            if modify == 'zbirth' and not self._validnlayers(proposalmodel) or not self._validmodel(proposalmodel):
                proposalmodel = None


        elif modify == 'rfnoise':
            proposalmodel = self.currentmodel
            proposalrfnoise = self._get_hyperparameter_proposal(modify)
            proposalswdnoise = self.currentswdnoise
            proposalvpvs = self.currentvpvs
            if not self._validrfnoise(proposalrfnoise):
                proposalmodel = None

        elif modify == 'swdnoise':
            proposalmodel = self.currentmodel
            proposalrfnoise = self.currentrfnoise
            proposalswdnoise = self._get_hyperparameter_proposal(modify)
            proposalvpvs = self.currentvpvs
            if not self._validswdnoise(proposalswdnoise):
                proposalmodel = None

        elif modify == 'vpvs':
            proposalmodel = self.currentmodel
            proposalrfnoise = self.currentrfnoise
            proposalswdnoise = self.currentswdnoise
            proposalvpvs = self._get_vpvs_proposal()
            if not self._validvpvs(proposalvpvs):
                proposalmodel = None
    

        return proposalmodel, proposalvpvs, proposalrfnoise, proposalswdnoise, self.dvs2, self.dra2, self.birthnucleus, self.ind_death
        
    def _validvpvs(self, vpvs):
        # only works if vpvs-priors is a range
        if vpvs < self.priors['vpvs'][0] or \
                    vpvs > self.priors['vpvs'][1]:
                return False
       
        return True
        
    def _get_vpvs_proposal(self):
        vpvs = copy.deepcopy(self.currentvpvs)
        vpvs_mod = self.rstate.normal(0, self.propdist[7])
        vpvs = vpvs + vpvs_mod
     
        return vpvs
    
    def _validswdnoise(self, noise):
        for idx in self.swdnoiseinds:
            if idx%2 == 0:
                idxn = 0
            else:
                idxn = 1
            if noise[idx] < self.swdnoisepriors[idxn][0] or \
                    noise[idx] > self.swdnoisepriors[idxn][1]:

                return False
        return True

    def _validrfnoise(self, noise):
        for idx in self.rfnoiseinds:
            if idx%2 == 0:
                idxn = 0
            else:
                idxn = 1


            if noise[idx] < self.rfnoisepriors[idxn][0] or \
                noise[idx] > self.rfnoisepriors[idxn][1]:
                return False

        return True

    def _get_hyperparameter_proposalidx(self, noise_array, counters):    
        # Calculate probabilities based on counters
        probabilities = counters / np.sum(counters)
        # Calculate the average count across all values
        average_count = np.mean(counters)
    
        # Set the threshold to the average count
        min_threshold = average_count
    
        # Check if any counter falls below the minimum threshold
        if np.any(counters < min_threshold):
            # Identify indices where counters are below the threshold
            low_count_indices = np.where(counters < min_threshold)[0]
        
            # Assign higher probabilities to those indices
            probabilities[low_count_indices] = 1.0 / len(low_count_indices)
        
        # Normalize probabilities to ensure they sum to 1
        probabilities /= np.sum(probabilities)
        # Choose a value based on probabilities
        chosen_index = self.rstate.choice(range(len(noise_array)), p=probabilities)
    
        # Update counters
        counters[chosen_index] += 1
        return counters, noise_array[chosen_index]

    def _get_hyperparameter_proposal(self, modify):
        'which noise should be chosen'
        propidx = PROPDIST[modify]  
        noise_mod = self.rstate.normal(0, self.propdist[propidx])

        if modify == 'rfnoise':
            noise = copy.deepcopy(self.currentrfnoise)
            self.rfcounters, ind = self._get_hyperparameter_proposalidx(self.rfnoiseinds, self.rfcounters)

        if modify == 'swdnoise':
            noise = copy.deepcopy(self.currentswdnoise)
            self.swdcounters, ind = self._get_hyperparameter_proposalidx(self.swdnoiseinds, self.swdcounters)
   
        noise[ind] = noise[ind] + noise_mod
        return noise




    def _validnlayers(self, model):
        vs, ra, vnoi = model[0], model[1], model[2]
        # check whether ncells lies within the prior
        ncellmin, ncellmax = self.priors['ncells']
        ncells = len(vs)
        if not (ncells >= ncellmin and ncells <= 5):
            return False
        return True 

    def _validmodel(self, model):
        """
        Check model before the forward modeling.
        - The model must contain all values > 0.
        """
        vs, ra, vnoi = model[0], model[1], model[2]

        # check whether ncells lies within the prior
        ncellmin, ncellmax = self.priors['ncells']
        ncells = len(vs)
        if not (ncells >= ncellmin and ncells <= ncellmax):

            return False
        # check whether vs lies within the prior
        vsmin = self.priors['vs'][0]
        vsmax = self.priors['vs'][1]
        if np.any(vs < vsmin) or np.any(vs > vsmax):

            return False

        # check whether vs lies within the prior
        if self.ramod:
            ramin = self.priors['ra'][0]
            ramax = self.priors['ra'][1]
            if np.any(ra < ramin) or np.any(ra > ramax):

                return False
        else:
            if np.any(ra != self.priors['ra']):

                return False


        # check whether interfaces lie within prio
        x = vnoi[:, 0]
        y = vnoi[:, 1]
        z = vnoi[:, 2]
        if np.any(x < self.grids['gridx'][0]) or np.any(x > self.grids['gridx'][1]):

            return False
        if np.any(y < self.grids['gridy'][0]) or np.any(y > self.grids['gridy'][1]):

            return False

        if np.any(z < self.grids.zmin) or np.any(z > self.grids.zmax):

            return False

        new_array = [tuple(row) for row in vnoi]
    
        uniques = np.unique(new_array, axis=0)
        if len(vnoi) !=  len(uniques):
            return False
   

        return True

    def _model_layerbirth(self, model, vspropidx, rapropidx):
        """
        Draw a random voronoi nucleus from model and assign a new Vs.
        The new Vs is based on the before Vs value at the drawn cell
        position (self.propdist[3]).
        """
        vs_vnoi, ra_vnoi, vnoi = model[0], model[1], model[2]

        # pick a random depth as a new nucleus
        nucleus_x = np.array(0.5*(self.grids.xmin+self.grids.xmax))
        nucleus_y = np.array(0.5*(self.grids.xmin+self.grids.xmax))
        nucleus_z = np.random.uniform(self.grids.zmin, self.grids.zmax)
        vnoi_birth = [nucleus_x, nucleus_y, nucleus_z]
        
        #use for update proposal kdtree
        self.birthnucleus = np.vstack((vnoi_birth)).T

        # find the closest nucleus
        kdtreedist, kdtreeidx = self.currentkdtree.query(vnoi_birth)
        vs_before = vs_vnoi[kdtreeidx]
        ra_before = ra_vnoi[kdtreeidx]

        vs_birth = vs_before + self.rstate.normal(0, self.propdist[vspropidx])
      
        vnoi_new = np.vstack((vnoi, np.array(vnoi_birth)))
        vs_new = np.concatenate((vs_vnoi, [vs_birth]))

        self.dvs2 = np.square(vs_birth - vs_before)

        if self.ramod:
            ra_birth = ra_before + \
                self.rstate.normal(0, self.propdist[rapropidx])
            ra_new = np.concatenate((ra_vnoi, [ra_birth]))
            self.dra2 = np.square(ra_birth - ra_before)
        else:
            ra_new = np.concatenate((ra_vnoi, [ra_before]))

        
        return [vs_new, ra_new, vnoi_new]


    def _model_cellbirth(self, model, vspropidx, rapropidx):
        """
        Draw a random voronoi nucleus from model and assign a new Vs.
        The new Vs is based on the before Vs value at the drawn cell
        position (self.propdist[3]).
        """
        vs_vnoi, ra_vnoi, vnoi = model[0], model[1], model[2]

        # pick a random point as a new nucleus
        # voronoi cells info
        nucleus_x = np.random.uniform(self.grids.xmin, self.grids.xmax)
        nucleus_y = np.random.uniform(self.grids.ymin, self.grids.ymax)
        nucleus_z = np.random.uniform(self.grids.zmin, self.grids.zmax)
        vnoi_birth = [nucleus_x, nucleus_y, nucleus_z]
        
        #use for update proposal kdtree
        self.birthnucleus = np.vstack((vnoi_birth)).T

        # find the closest nucleus
        kdtreedist, kdtreeidx = self.currentkdtree.query(vnoi_birth)
        vs_before = vs_vnoi[kdtreeidx]
        ra_before = ra_vnoi[kdtreeidx]

        vs_birth = vs_before + self.rstate.normal(0, self.propdist[vspropidx])

        vnoi_new = np.vstack((vnoi, np.array(vnoi_birth)))
        vs_new = np.concatenate((vs_vnoi, [vs_birth]))

        self.dvs2 = np.square(vs_birth - vs_before)

        if self.ramod:
            ra_birth = ra_before + \
                self.rstate.normal(0, self.propdist[rapropidx])
            ra_new = np.concatenate((ra_vnoi, [ra_birth]))
            self.dra2 = np.square(ra_birth - ra_before)
        else:
            ra_new = np.concatenate((ra_vnoi, [ra_before]))
        
        return [vs_new, ra_new, vnoi_new]

    def _model_celldeath(self, ind, model, vspropidx, rapropidx):
        """
        Remove a random voronoi cell from model. Delete corresponding
        Vs from model.
        """
        vs_vnoi, ra_vnoi, vnoi = model[0], model[1], model[2]
        self.ind_death = ind
        vnoi_before = vnoi[ind]
        vs_before = vs_vnoi[ind]
        ra_before = ra_vnoi[ind]

        vnoi_new = np.delete(vnoi, (ind), axis=0)
        vs_new = np.delete(vs_vnoi, ind)
        ra_new = np.delete(ra_vnoi, ind)

        nucleus = copy.deepcopy(vnoi_new)
        kdtree_death = KDTree(nucleus)
        kdtreedist, kdtreeidx = kdtree_death.query(vnoi_before)

        vs_after = vs_new[kdtreeidx]
        ra_after = ra_new[kdtreeidx]

        self.dvs2 = np.square(vs_after - vs_before)
        self.dra2 = np.square(ra_after - ra_before)



        return [vs_new, ra_new, vnoi_new]

    def _model_vschange(self, ind, model, propidx):
        """Randomly chose a cell to change Vs with Gauss distribution."""
        vs_mod = self.rstate.normal(0, self.propdist[propidx])
        model[0][ind] = model[0][ind] + vs_mod
        return model

    def _model_rachange(self, ind, model, propidx):
        """Randomly chose a cell to change Vs with Gauss distribution."""
        ra_mod = self.rstate.normal(0, self.propdist[propidx])
        model[1][ind] = model[1][ind] + ra_mod
        return model

    def _model_vnoi_move(self, ind, model, propidx):
        """Randomly chose a cell to change (x_vnoi, y_vnoi, z_vnoi) with Gauss distribution."""
        self.vnoi_move_ind = ind

        # Randomly choose a direction to change!!!!!
        ind_dir = self.rstate.randint(0, 3)
        vnoi_mod = self.rstate.normal(0, self.propdist[propidx])

        # Use different proposal distribution for shollow and deep voronoi cell
        if model[2][ind][2] > ((self.grids.zmax+self.grids.zmin)/2):
            propidx = PROPDIST['vmod2']
            vnoi_mod = self.rstate.normal(0, self.propdist[propidx])
            self.vnoi_move2 = True
        else:
            vnoi_mod = self.rstate.normal(0, self.propdist[propidx])

        model[2][ind][ind_dir] = model[2][ind][ind_dir] + vnoi_mod
        self.vnoi_move_cell = np.vstack((model[2][ind])).T
        return model

    def _model_zvnoi_move(self, ind, model, propidx):
        """Randomly chose a cell to change (z_vnoi) with Gauss distribution."""
        self.vnoi_move_ind = ind

        # Randomly choose a direction to change!!!!!
        ind_dir = 2 #self.rstate.randint(0, 3)

        if model[2][ind][2] > ((self.grids.zmax+self.grids.zmin)/2):
            propidx = PROPDIST['vmod2']
            vnoi_mod = self.rstate.normal(0, self.propdist[propidx])
            self.vnoi_move2 = True
            
        else:
            vnoi_mod = self.rstate.normal(0, self.propdist[propidx])

        model[2][ind][ind_dir] = model[2][ind][ind_dir] + vnoi_mod
        self.vnoi_move_cell = np.vstack((model[2][ind])).T
        return model
    
    def _get_modelproposal(self, modify):
        model = copy.deepcopy(self.currentmodel)


        ind = self.rstate.randint(0, model[0].size) 
        if modify == 'vsmod':
            propidx = PROPDIST[modify]
            propmodel = self._model_vschange(ind,model, propidx)
        if modify == 'ramod':
            propidx = PROPDIST[modify]
            propmodel = self._model_rachange(ind,model, propidx)
        elif modify == 'vmod':
            propidx = PROPDIST[modify]
            propmodel = self._model_vnoi_move(ind,model, propidx)
        elif modify == 'zvmod':
            propidx = PROPDIST['vmod']
            propmodel = self._model_zvnoi_move(ind,model, propidx)
        elif modify == 'zbirth':
            vspropidx = PROPDIST['vsbirth/death']
            rapropidx = PROPDIST['rabirth/death']
            propmodel = self._model_layerbirth(model, vspropidx, rapropidx)
        elif modify == 'birth':
            vspropidx = PROPDIST['vsbirth/death']
            rapropidx = PROPDIST['rabirth/death']
            propmodel = self._model_cellbirth(model, vspropidx, rapropidx)
        elif modify == 'death':
            vspropidx = PROPDIST['vsbirth/death']
            rapropidx = PROPDIST['rabirth/death']
            propmodel = self._model_celldeath(ind,model, vspropidx, rapropidx)
        return propmodel

class RecursiveSampleMoments:

    """Iteratively constructs a sample mean and covariance, given input
    samples. Used to capture an estimate of the mean and covariance of the bias
    of an MLDA coarse model, and for the Adaptive Metropolis (AM) proposal.
    Attributes
    ----------
    mu : numpy.ndarray
        The mean array.
    sigma : numpy.ndarray
        The covariance matrix.
    d : int
        The dimensionality.
    t : int
        The sample size, i.e. iteration counter.
    sd : float
        The AM scaling parameter.
    epsilon : float
        Parameter to prevent C from becoming singular (used for AM).
    Methods
    ----------
    get_mu()
        Returns the current mean.
    get_sigma()
        Returns the current covariance matrix.
    update(x)
        Update the sample moments with an input array x.
    """

    def __init__(self, mu0, sigma0, npairs=0, t=1, sd=1, epsilon=0):
        """
        Parameters
        ----------
        mu0 : numpy.ndarray
            The initial mean array.
        sigma0 : numpy.ndarray
            The initial covariance matrix.
        t : int, optional
            The initial sample size, i.e. iteration counter. Default is 1.
        sd : float, optional
            The AM scaling parameter. Default is 1
        epsilon : float, optional
            Parameter to prevent C from becoming singular (used for AM).
            Default is 0.
        """

        # set the initial mean and dimensionality
        self.mu = mu0
        self.d = self.mu[0].shape[0]

        # set the initial covariance matrix.
        self.sigma = sigma0

        # set the counter
        self.t = t

        # set AM-specific parameters.
        self.sd = sd
        self.epsilon = epsilon

        # set number of pairs:
        self.npairs = npairs
        #print ('NPARIES', self.npairs )

    def __call__(self):
        """
        Returns
        ----------
        tuple
            Returns tuple of the current (mean, covariance), each a numpy.ndarray.
        """

        return self.mu, self.sigma

    def get_mu(self):
        """
        Returns
        ----------
        numpy.ndarray
            Returns the current mean.
        """

        # Returns the current mu value
        return self.mu

    def get_sigma(self):
        """
        Returns
        ----------
        numpy.ndarray
            Returns the current covariance.
        """

        # Returns the current covariance value
        return self.sigma

    def update(self, x):
        """
        Parameters
        ----------
        x : numpy.ndarray
            Updates the sample moments using an input array x.
        """

        # Updates the mean and covariance given a new sample x
        mu_previous = copy.copy(self.mu)
        diff =  mu_previous-x
        #print ('SAMPLING difff',  mu_previous[:,0], x[:,0], diff[:,0])
        self.mu = (1 / (self.t + 1)) * (self.t * mu_previous + x)
        #print (self.mu[:,0] )
        for n in range(self.npairs):
            self.sigma[n] = (self.t - 1) / self.t * self.sigma[n] + self.sd / self.t * (
                self.t * np.sum(mu_previous[n] **2)
                - (self.t + 1) * np.sum(self.mu[n]**2)
                + np.sum( x[n]**2)
                + self.epsilon 
            )
        #    print ('T', self.t, np.sum(mu_previous[n] **2),  np.sum(self.mu[n]**2) , np.sum( x[n]**2))
        #print (self.sigma, mu_previous.shape, self.mu.shape, x.shape)
        #self.sigma = (self.t - 1) / self.t * self.sigma + self.sd / self.t * (
        #    self.t * np.outer(mu_previous, mu_previous)
        #    - (self.t + 1) * np.outer(self.mu, self.mu)
        #    + np.outer(x, x)

        #)    
        self.t += 1



