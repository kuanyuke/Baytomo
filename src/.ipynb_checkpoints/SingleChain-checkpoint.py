#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:14:37 2022

@author: kuanyu
"""

import sys
import copy
import time as gettime
import numpy as np
import os.path as op
import os
from scipy.spatial import cKDTree as KDTree
import scipy.spatial.distance
import pickle
from Baytomo.Models import Model  
from Baytomo.Sampling import Sampling
from pathlib import Path
from Baytomo import utils
import scipy
import logging
import math
import pickle
logger = logging.getLogger()

PAR_MAP = {'vsmod': 0, 'ramod': 1, 'vmod': 2, 'birth': 3, 'rabirth': 4, 'death': 3, 
            'swdnoise': 5, 'rfnoise': 6, 'vpvs': 7, "vmod2":8 }

rf_targets = ['prf', 'srf']
swd_targets = ['rdispph', 'ldispph', 'rdispgr', 'ldispgr']



class SingleChain(object):
    def __init__(self, targets, grids=dict(), initparams=dict(), modelpriors=dict(),
                 existdatapath=False):
        """
        Initialize a SingleChain instance for Bayesian inversion.

        Parameters:
            targets (YourTargetsClass): An instance of the class containing inversion targets.
            grids (dict): Dictionary containing grid-related parameters.
            initparams (dict): Dictionary containing initialization parameters.
            modelpriors (dict): Dictionary containing prior information about the model.
            existdatapath (str): Path to existing data for resuming or initializing the chain.

        Note: You can refer to your existing parameters to understand what each parameter represents.
        """

        defaults = utils.get_path('defaults.ini')
        self.grids, self.priors, self.initparams = utils.load_params(defaults)
        

        self.grids.update(grids)
        self.initparams.update(initparams)
        self.priors.update(modelpriors)
        self.station = self.initparams.get('station')

        # Metrolopis rule
        self.dv = (self.priors['vs'][1] - self.priors['vs'][0])
        raprior = self.priors['ra']
        if type(raprior) in [int, float, np.float64]:
            self.ramod = False
            self.dra = 0
        else:
            self.ramod = True
            self.dra = (raprior[1] - raprior[0])

        # set targets and inversion specific parameters
        self.targets = targets
        if self.targets.nswdtargets != 0:
            self.swd = True
        else:
            self.swd = False
            self.currentswdnoise = np.ones(2) * np.nan

        if self.targets.nrftargets != 0:
            self.rf = True
        else:
            self.rf = False
            self.currentrfnoise = np.ones(2) * np.nan
        
        
        # grid info
        self.grids = Model.create_grid_model(self.grids)
        self.init_forward_modelling()
        self.gridsmodel = self.grids.gridsmodel

        # mcmc info
        self.nchains = self.initparams.get('nchains')

        self.iter_phase1 = int(self.initparams['iter_burnin'])
        self.iter_phase2 = int(self.initparams['iter_main'])
        self.iiter = -self.iter_phase1
        self.lastmoditer = self.iiter
        self.iterations = self.iter_phase1 + self.iter_phase2
        self.thinning = int(self.initparams['thinning'])
        self.propdist = np.array(self.initparams['propdist'])
        self.acceptance = self.initparams['acceptance']
        self.n = 0

        # save file for offline-plotting
        savepath = op.join(self.initparams['savepath'], 'data')
        outfile = op.join(savepath, '%s_config.pkl' % self.station)
        Path(savepath).mkdir(parents=True, exist_ok=True)
        self.extract_fromlastmodel_path = existdatapath
        if not op.exists(savepath) :
            utils.save_config(self.targets, outfile, grids=self.grids, priors=self.priors,
                              initparams=self.initparams)
            self.extract_fromlastmodel_path = False
            if existdatapath:
                self.extract_fromlastmodel_path =  existdatapath
        elif not op.exists(outfile):
            utils.save_config(self.targets, outfile, grids=self.grids, priors=self.priors,
                              initparams=self.initparams)
            self.extract_fromlastmodel_path = False
        elif op.exists(savepath) and existdatapath:
            # if exists, sample from the last model
            self.extract_fromlastmodel_path = savepath
   
                            

    def append_currentmodel2(self):
        '''
        Append a new array to the file
        Note that this will not change the header
        '''
        # Define the save path based on initialization parameters
        savepath = op.join(self.initparams['savepath'], 'data')

        # Determine the phase based on the current iteration index
        if self.iiter < 0:
            phase =1
        else:
            phase = 2
        # Construct the file name         
        fname =  op.join(savepath, 'c%.3d_p%smodelsvon.txt' % (self.chainidx, phase))

        # Append the current model's von array to the file
        with open(fname, "a") as fh:
                np.savetxt(fh, self.currentmodel[2].T.reshape((3,-1)), fmt="%s", header=str(self.currentmodel[2].shape))

        
        
    def append_currentmcvalues(self):
        '''
        Append current Markov chain values to files
        '''
        # Helper function to write data to a file
        def write(datatype, data):
            savepath = op.join(self.initparams['savepath'], 'data')
            if self.iiter < 0:
                phase =1
            else:
                phase = 2

            # Construct the file name
            fname = op.join(savepath, 'c%.3d_p%s%s.txt' % (self.chainidx, phase, datatype))

            # Append the data to the file
            with open(fname, "a") as fh:
                np.savetxt(fh, data, newline=" \n")
        
        # Prepare current Markov chain values
        currentvalues = [self.currentmodel[0], self.currentmodel[1], self.currentvpvs,self.currentmisfits,\
        self.currentlikelihood, self.currentswdnoise, self.currentrfnoise]

        # Define names corresponding to each value
        names = ['modelsvs', 'modelsra','vpvs','misfits', 'likes', 'swdnoise', 'rfnoise']

        # Iterate over values and write them to respective files
        for i, name in enumerate(names):
            write(name, [currentvalues[i]])
        

            
    def init_forward_modelling(self):
        for n, target in enumerate(self.targets.targets):
            if target.ref in rf_targets:
                stasloc = list(target.obsdata.stas.values())
                self.staingrididx = Model.sta_nearst_grid(stasloc, self.grids)
                target.moddata.plugin.init_grids_model(self.grids,
                                                       self.staingrididx)
            if target.ref in swd_targets:
                target.moddata.plugin.init_grids_model(self.grids)

    def draw_init_model(self):
        "Draw an intial model and noise from unifrom distribution"
        imodel = self.sampling.draw_initmodel(self.grids)
        ivpvs = self.sampling.draw_initvpvs(len(imodel[0]))
        irfnoise, rfcorrfix = self.sampling.draw_init_rfnoiseparams(self.targets)
        iswdnoise, swdfracfix = self.sampling.draw_init_swdnoiseparams(self.targets)
        rcond = self.initparams['rcond']
        self.sampling.set_rftarget_covariance(self.targets,rfcorrfix, irfnoise, rcond)
        return imodel, iswdnoise, irfnoise, ivpvs



    def _init_model_and_currentvalues(self):
        "Draw an intial model and noise from prior distribution"
        def draw_init_model():
            #print ("IINIT model")
            imodel, iswdnoise, irfnoise, ivpvs = self.draw_init_model()
            return imodel, iswdnoise, irfnoise, ivpvs
        
        def draw_init_gridmodel():
            imodel, iswdnoise, irfnoise, ivpvs = draw_init_model()

            # kdtree to grid: each grid has a kdtreeidx number corresponding to the current nuclei
            # 1st: calculate kdtree based on location of nuclei
            # vor = Voronoi(imodel[2])
            # upscaling

            nucleus = copy.deepcopy(imodel[2])
            kdtree = KDTree(nucleus)

            ##check if nuclei distance > grid distance
            kdtreedist, kdtreeidx = kdtree.query(nucleus, k=2)
            invondist = kdtreedist[:,1]
            while np.any(invondist< self.grids.mindist):
                imodel, iswdnoise, irfnoise, ivpvs =  draw_init_model()
                nucleus = copy.deepcopy(imodel[2])
            
                kdtree = KDTree(nucleus)                
                kdtreedist, kdtreeidx = kdtree.query(nucleus, k=2)
                invondist = kdtreedist[:,1]

           
            vs = imodel[0]
            vp = imodel[0] * ivpvs
            ra = imodel[1]
            
            # 2nd find the closest nuclei to each grid so each grid would get parameters (vs,ra)
            # based on the close nuclei
            kdtreedist, kdtreeidx = kdtree.query(self.gridsmodel)
            gridvon, gridvp, gridvs, gridra = Model.kdtree_to_grid(nucleus=nucleus,vp=vp, vs=vs, ra=ra,
                                                      kdtreeidx=kdtreeidx)
            igridmodel = [gridvs, gridra]


            self.targets.evaluate(von = gridvon, vp=gridvp, vs=gridvs, ra=gridra, swdnoise=iswdnoise,
                              rfnoise=irfnoise)
            
              
            return imodel, igridmodel, iswdnoise, irfnoise, ivpvs, kdtree, \
                    kdtreeidx, kdtreedist


        imodel, igridmodel, iswdnoise, irfnoise, ivpvs, kdtree, kdtreeidx, kdtreedist = draw_init_gridmodel()
        while np.isnan(self.targets.proposallikelihood):
            imodel, igridmodel, iswdnoise, irfnoise, ivpvs, kdtree, kdtreeidx, kdtreedist = draw_init_gridmodel()

        self.n = 0  # accepted models counter
        self.accept_as_currentmodel(
            imodel, igridmodel, iswdnoise, irfnoise, ivpvs, kdtree, kdtreeidx, kdtreedist)
        self.targets.accept_as_currentmoddata()
        self.init_kdtreeidx = kdtreeidx
        self.init_kdtreedist = kdtreedist
        self.append_currentmcvalues()
        self.append_currentmodel2()

        
    def _init_existmodel(self, imodel,iswdnoise, irfnoise, ivpvs):
        "Calculate the current model based on exist model"
        if math.isnan(irfnoise[0]):
            irfnoise, rfcorrfix = self.sampling.draw_init_rfnoiseparams(self.targets)
        else:
            _, rfcorrfix = self.sampling.draw_init_rfnoiseparams(self.targets)
        if math.isnan(iswdnoise[0]):
            iswdnoise, swdfracfix = self.sampling.draw_init_swdnoiseparams(self.targets)
        else:
            _, swdfracfix = self.sampling.draw_init_swdnoiseparams(self.targets)


        rcond = self.initparams['rcond']
        self.sampling.set_rftarget_covariance(self.targets,rfcorrfix, irfnoise, rcond)
        
        
        nucleus = copy.deepcopy(imodel[2])
        vs =  copy.deepcopy(imodel[0])
        vp =  copy.deepcopy(imodel[0]) *  copy.deepcopy(ivpvs)
        ra =  copy.deepcopy(imodel[1])
        kdtree = KDTree(nucleus)
        kdtreedist, kdtreeidx = kdtree.query(self.gridsmodel)
  
        gridvon, gridvp, gridvs, gridra = Model.kdtree_to_grid(nucleus=nucleus,vp=vp, vs=vs, ra=ra,
                                                      kdtreeidx=kdtreeidx)
        igridmodel = [gridvs, gridra]
        self.targets.evaluate(von = gridvon, vp=gridvp, vs=gridvs, ra=gridra, swdnoise=iswdnoise,
                              rfnoise=irfnoise)

        self.n = 0  # accepted models counter
        self.accept_as_currentmodel(
            imodel, igridmodel, iswdnoise, irfnoise, ivpvs, kdtree, kdtreeidx, kdtreedist)
        self.targets.accept_as_currentmoddata()
        self.init_kdtreeidx = kdtreeidx
        self.init_kdtreedist = kdtreedist
        self.append_currentmcvalues()
        self.append_currentmodel2()

    def set_rftarget_covariance(self, corrfix, inoise, rcond=None):
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
        for i, target in enumerate(self.targets.rftargets):
            # if target.noiseref == 'swd':
            #    if np.any(np.isnan(target.obsdata.yerr)):
            # diagonal for each target, corr inrelevant for likelihood, rel error
            # target.get_covariance = target.valuation.get_covariance_nocorr
            #        continue
            #    else:
            # diagonal for each target, corr inrelevant for likelihood
            # target.get_covariance = target.valuation.get_covariance_nocorr_scalederr
            #       continue

            # elif  target.noiseref == 'rf':
            #   target_corrfix = corrfix[::2]
            #   target_noise_corr = inoise[::2]
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

    def _valid_dist(self, kdtree, nucleus):
        #check if nuclei distance > grid distance
        kdtreedist, kdtreeidx = kdtree.query(nucleus, k=2)
        invondist = kdtreedist[:,1]
        if np.any(invondist < self.grids.mindist):
                return False
        return True            

    def adjust_propdist(self):
        """
        Modify self.propdist to adjust acceptance rate of models to given
        percentace span: increase or decrease by five percent.
        """
        with np.errstate(invalid='ignore'):
            acceptrate = self.accepted / self.proposed * 100

        # minimum distribution width forced to be not less than 1 m/s, 1 m
        # actually only touched by vs distribution
        propdistmin = np.full(acceptrate.size, 0.001)

        for i, rate in enumerate(acceptrate):
            if np.isnan(rate) or rate < 0:
                continue
            if rate < self.acceptance[0]:
                new = self.propdist[i] * 0.95
                if new < propdistmin[i]:
                    new = propdistmin[i]
                self.propdist[i] = new

            elif rate > self.acceptance[1]:
                self.propdist[i] = self.propdist[i] * 1.05
            else:
                pass

        self.sampling.update_prodist(self.propdist)
     

                
    def get_acceptance_probability2(self, modify):
        """
        Acceptance probability will be computed dependent on the modification.
        Parametrization alteration (Vs or voronoi nuclei position)
            the acceptance probability is equal to likelihood ratio.
        Model dimension alteration (cell birth or death)
            the probability was computed after the formulation of Bodin et al.,
            2012: 'Transdimensional inversion of receiver functions and
            surface wave dispersion'.
        """
        if modify in ['vsmod', 'ramod', 'vmod', 'zvmod','swdnoise', 'rfnoise', 'vpvs']:
            # only velocity or thickness changes are made
            # also used for noise changes
            alpha = self.targets.proposallikelihoodjoint - self.currentlikelihoodjoint

        elif modify in ['birth',  'zbirth']:
            theta_vs = self.propdist[3]  # Gaussian distribution
            theta_ra = self.propdist[4]  # Gaussian distribution)
            if self.ramod:
                A = (theta_vs * np.sqrt(2 * np.pi)) / self.dv
                B = (theta_ra * np.sqrt(2 * np.pi)) / self.dra
                C = self.dvs2 / (2. * np.square(theta_vs))
                D = self.dra2 / (2. * np.square(theta_ra))
                E = self.targets.proposallikelihood - self.currentlikelihoodjoint
                alpha = np.log(A) + np.log(B) + C + D + E
            else:
                A = (theta_vs * np.sqrt(2 * np.pi)) / self.dv
                B = self.dvs2 / (2. * np.square(theta_vs))
                C = self.targets.proposallikelihood - self.currentlikelihoodjoint
                alpha = np.log(A) + B + C
                
        elif modify in ['death', ]:
            theta_vs = self.propdist[3]  # Gaussian distribution
            theta_ra = self.propdist[4]  # Gaussian distribution

            if self.ramod:
                A = self.dv / (theta_vs * np.sqrt(2 * np.pi))
                C = self.dra / (theta_ra * np.sqrt(2 * np.pi))
                B = self.dvs2 / (2. * np.square(theta_vs))
                D = self.dra2 / (2. * np.square(theta_ra))
                E = self.targets.proposallikelihood - self.currentlikelihoodjoint
                alpha = np.log(A) + np.log(B) - C - D + E
            else:
                A = self.dv / (theta_vs * np.sqrt(2 * np.pi))
                B = self.dvs2 / (2. * np.square(theta_vs))
                C = self.targets.proposallikelihood - self.currentlikelihoodjoint
                alpha = np.log(A) - B + C
        return alpha

    def get_acceptance_probability(self, modify):
        """
        Acceptance probability will be computed dependent on the modification.
        Parametrization alteration (Vs or voronoi nuclei position)
            the acceptance probability is equal to likelihood ratio.
        Model dimension alteration (cell birth or death)
            the probability was computed after the formulation of Bodin et al.,
            2012: 'Transdimensional inversion of receiver functions and
            surface wave dispersion'.
        """

        if modify in ['vsmod', 'ramod', 'vmod', 'vmod2', 'zvmod','swdnoise', 'rfnoise', 'vpvs']:
            # only velocity or thickness changes are made
            # also used for noise changes
            alpha = self.targets.proposallikelihood - self.currentlikelihood
        elif modify in ['birth',  ]:
            theta_vs = self.propdist[3]  # Gaussian distribution
            theta_ra = self.propdist[4]  # Gaussian distribution

            if self.ramod:
                A = (theta_vs * np.sqrt(2 * np.pi)) / self.dv
                B = (theta_ra * np.sqrt(2 * np.pi)) / self.dra
                C = self.dvs2 / (2. * np.square(theta_vs))
                D = self.dra2 / (2. * np.square(theta_ra))
                E = self.targets.proposallikelihood - self.currentlikelihood
                alpha = np.log(A) + np.log(B) + C + D + E
            else:
                A = (theta_vs * np.sqrt(2 * np.pi)) / self.dv
                B = self.dvs2 / (2. * np.square(theta_vs))
                C = self.targets.proposallikelihood - self.currentlikelihood
                alpha = np.log(A) + B + C
                
                
        elif modify in ['death', ]:
            theta_vs = self.propdist[3]  # Gaussian distribution
            theta_ra = self.propdist[4]  # Gaussian distribution
            if self.ramod:
                A = self.dv / (theta_vs * np.sqrt(2 * np.pi))
                C = self.dra / (theta_ra * np.sqrt(2 * np.pi))
                B = self.dvs2 / (2. * np.square(theta_vs))
                D = self.dra2 / (2. * np.square(theta_ra))
                E = self.targets.proposallikelihood - self.currentlikelihood
                alpha = np.log(A) + np.log(B) - C - D + E
            else:
                A = self.dv / (theta_vs * np.sqrt(2 * np.pi))
                B = self.dvs2 / (2. * np.square(theta_vs))
                C = self.targets.proposallikelihood - self.currentlikelihood
                alpha = np.log(A) - B + C
        return alpha

    def accept_as_currentmodel(self, model, gridmodel,swdnoise, rfnoise, vpvs, kdtree, kdtreeidx, kdtreedist):
        """Assign currentmodel and currentvalues to self."""
        self.currentmisfits = self.targets.proposalmisfits
        self.currentlikelihood = self.targets.proposallikelihood
        self.currentlikelihoodjoint  = self.targets.proposallikelihoodjoint

        self.currentmodel = model
        self.currentgridmodel = gridmodel

        if self.swd:
            self.currentswdnoise = swdnoise
        if self.rf:
            self.currentrfnoise = rfnoise
        self.currentvpvs = vpvs
        self.lastmoditer = self.iiter
        self.currentkdtree = kdtree
        self.currentkdtreeidx = kdtreeidx
        self.currentkdtreedist = kdtreedist

    def cal_dist_von2grid(self, modify,nucleus ):
        if modify in ['vmod', 'zvmod', 'vmod2', 'birth','death'] or self.iiter == -self.iter_phase1:
            # Calculate the KDTree based on the nuclei of the proposed model
            proposalkdtree = KDTree(nucleus)
            # Find the closest nuclei to each grid so each grid would get parameters (vs,ra)
            # based on the close nuclei
            proposalkdtreedist, proposalkdtreeidx = proposalkdtree.query(self.gridsmodel)
            
        elif modify in ['birth','zbirth']:
            # Calculate the KDTree based on the nuclei of the proposed model
            proposalkdtree = KDTree(nucleus)
            # Check if the calculated distance is valid
            if not self._valid_dist( proposalkdtree, nucleus):
                return None, None, None
            
            # Copy current KDTree index and distance
            proposalkdtreeidx = copy.copy(self.currentkdtreeidx)
            proposalkdtreedist = copy.copy(self.currentkdtreedist)
            
            # Calculate the distance for each grid point to the birth nucleus
            proposalkdtreedist, proposalkdtreeidx = proposalkdtree.query(self.gridsmodel)
            birth_min_dists = np.sqrt(((self.gridsmodel-self.birthnucleus)**2).sum(axis=1))  # compute distances

            # Compare the difference between the distance for each grid point to the birth nucleus and
            # the distance for each grid point to all nuclei in the current Voronoi model
            # If negative, it means that the grid is closer to the new birth nucleus than the old one, so replace
            dist_diff = birth_min_dists - proposalkdtreedist
            proposalkdtreeidx[np.where(dist_diff< 0)] = len(nucleus)-1 
            proposalkdtreedist[np.where(dist_diff< 0)] = birth_min_dists[np.where(dist_diff< 0)]


                
        elif modify in ['death']:
            # Calculate the KDTree based on the nuclei of the proposed model
            proposalkdtree = KDTree(nucleus)

            # Check if the calculated distance is valid
            if not self._valid_dist( proposalkdtree, nucleus):
                return None, None, None

            # Copy current KDTree index and distance
            proposalkdtreeidx = copy.copy(self.currentkdtreeidx)
            proposalkdtreedist = copy.copy(self.currentkdtreedist)
            
            # Find the indices where the proposed KDTree index is equal to the index of death
            idx = np.where((proposalkdtreeidx ==self.ind_death))[0]
            deathidxgrid =  self.gridsmodel[idx]

            # Calculate the distance between the grid points with the same index and the nuclei
            death_dist = scipy.spatial.distance.cdist(deathidxgrid, nucleus)
            death_min_dist_idx = np.argmin(death_dist, axis=1)
            death_min_dists = np.min(death_dist, axis=1)
            
            # Update KDTree index and distance for death modification
            proposalkdtreeidx =  np.where(proposalkdtreeidx>self.ind_death, proposalkdtreeidx-1, proposalkdtreeidx)
            proposalkdtreeidx[idx] = death_min_dist_idx
            proposalkdtreedist[idx] = death_min_dists

        elif modify in ['vsmod', 'ramod'] + self.vpvsmods + self.noisemods:
            # The neuclei don't chage in these modificatoin, so use the current one 
            proposalkdtree = copy.copy(self.currentkdtree)
            proposalkdtreeidx = copy.copy(self.currentkdtreeidx)
            proposalkdtreedist = copy.copy(self.currentkdtreedist)

        if proposalkdtree is not None and proposalkdtreeidx is not None and proposalkdtreedist is not None:
            # Unpack values
            return proposalkdtree, proposalkdtreeidx, proposalkdtreedist



    def iterate(self):
        """
        Perform one iteration of the Bayesian inversion.

        This method implements two strategies where the inversion process is divided into different phases:

        First strategy
        1. In the first one percent, the model dimension is fixed. 
        2. In the first twenty percent, only run SWD, free dimenison
        3. In the first thirty percent, fix the SWD noise, RF joint inversion
        4. After the first thirty percent, FREE every parameters 

        Second strategy
        1.  In the first one percent, the model dimension is fixed.
        2.  In the first twenty percent, the noise of each data  is fixed to one variable.
        3.  Free all the noise parameters
        """

        # Constants for phase percentages
        if  self.iiter < (-self.iter_phase1 + (self.iter_phase1 * 0.01)):

            modifications=['vsmod', 'zvmod', 'zbirth', 'death']  + self.vpvsmods 
           
            #if  len(self.targets.targets )== 1:
            modifications += self.noisemods

        #elif  self.iiter < (-self.iter_phase1 + (self.iter_phase1 * 0.03)) and len(self.targets.targets ) >  1:
        #        modifications = self.modelmods +self.dimmods  + self.vpvsmods 

        else:   
            modifications =self.modifications


        modify = self.rstate.choice( modifications)

        # Propose a new model and noise parameters based on the chosen modification         
        proposalmodel, proposalvpvs, proposalrfnoise, proposalswdnoise, self.dvs2, self.dra2, self.birthnucleus, self.ind_death=\
        self.sampling.sampling(modify, self.currentmodel, self.currentvpvs, self.currentrfnoise, self.currentswdnoise, self.currentkdtree)


        # If a valid proposal model and noise parameters are not found, exit the iteration
        if proposalmodel is None:
            # If not a valid proposal model and noise params are found,
            # leave self.iterate and try with another modification
            # should not occur often.         
            return

        
        if self.sampling.vnoi_move2:
            modify = 'vmod2'
        if modify == 'zvmod':
            modify = 'vmod'
        if modify == 'zbirth':
            modify = 'birth'

        if modify != 'death':
            paridx = PAR_MAP[modify]   
            self.proposed[paridx] += 1 
        
 
        # Compute synthetic data and likelihood, misfit for the proposed model
        vs = copy.deepcopy(proposalmodel[0])
        vp = copy.deepcopy(proposalmodel[0]) * proposalvpvs
        ra = proposalmodel[1]

        # kdtree to grid: each grid has a kdtreeidx number corresponding to the current nuclei
        # 1st: calculate kdtree based on location of nuclei
        # Calculate kdtree based on the location of nuclei in the proposed model
        nucleus = copy.deepcopy(proposalmodel[2])
  
        proposalkdtree, proposalkdtreeidx, proposalkdtreedist = self.cal_dist_von2grid(modify, nucleus)
        # If kdtree is not calculated successfully, exit the iteration
        if proposalkdtree is None:
            return
            
   
        # Convert kdtree to grid for the proposed model
        gridvon, gridvp, gridvs, gridra = Model.kdtree_to_grid(nucleus=nucleus,vp=vp, vs=vs, ra=ra,
                                                      kdtreeidx=proposalkdtreeidx)
        # Create a grid model for the proposed model
        proposalgridmodel = [gridvs, gridra]
      
        # Evaluate synthetic data for the proposed model
        self.targets.evaluate(von = gridvon, vp=gridvp, vs=gridvs, ra=gridra, swdnoise=proposalswdnoise,
                              rfnoise=proposalrfnoise)

        # If two consecutive bad models are produced, exit the iteration
        if np.isnan(self.targets.proposallikelihood):
            logger.debug('iiter %s  Not able to calculate Forward modelling \n' % (self.chainidx))

            return     
     
        # Calculate acceptance probability for the proposed modification.
        # these are log values ! alpha is log.
        u = np.log(self.rstate.uniform(0, 1))
        alpha = self.get_acceptance_probability(modify)

        # Accept the proposed model with probability alpha, or reject it with probability (1 - alpha)    
        if u < alpha:

            self.accept_as_currentmodel(
                proposalmodel, proposalgridmodel, proposalswdnoise, proposalrfnoise, 
                proposalvpvs, proposalkdtree, proposalkdtreeidx, proposalkdtreedist)
            # if the model is accepted, then the proposed model in forward calculation will replace current model
            # however, if the forward calculation has not be calculate last time due to no noise modification or something
            # the propose forward calculation won't be accepted because the propose one is the one beforehand
            #if  self.targets.targets[1].moddata.plugin.cacl and self.targets.targets[0].moddata.plugin.cacl:
            self.targets.accept_as_currentmoddata()
            if modify != 'death':
                self.accepted[paridx] += 1

            # Increment the acceptance counter
            self.n += 1

        # stabilize model acceptance rate
        if self.iiter % 1000 == 0:
            #print(f'{self.chainidx}  iiter {self.iiter} ALL PROPOSED [ {", ".join(map(str, self.proposed))}]')
            #print(f'{self.chainidx} iiter {self.iiter}  ACCEPT [ {" ".join(map(str, self.accepted))} ]')
            #print(f'{self.chainidx} iiter {self.iiter} propdist {self.propdist}')
            if 'zvmod' in modifications:
                index = modifications.index('zvmod')
                modifications[index] = 'vmod'

            if 'zbirth' in modifications:
                index = modifications.index('zbirth')
                modifications[index] = 'birth'

            if  all(self.proposed[PAR_MAP[mod]] != 0 for mod in modifications):
                self.adjust_propdist()

       

    def run_chain(self, chainidx, random_seed=None,  phase=1):
        """
        Run the Bayesian inversion chain.

        Parameters:
            chainidx (int): Index of the chain.
            random_seed (int): Seed for the random number generator.

        Note: You can refer to your existing code to understand the purpose of each parameter.
        """

        self.chainidx = chainidx
        self.rstate = np.random.RandomState(random_seed)
        self.iiter = -self.iter_phase1 - 1
        self.save_nmodels = 1 
        
        self.accepted = np.zeros(len( self.propdist))
        self.proposed = np.zeros(len( self.propdist))


         # Initialize sampling function
        self.sampling=Sampling(self.chainidx,  self.priors, self.grids, self.propdist, swd = self.swd, rf=self.rf, ramod = self.ramod, rstate = self.rstate)

        # Initialize starting model, either from existing model or ramdoly sample 
        tnull0 = gettime.time()
        if self.extract_fromlastmodel_path:

            imodel,iswdnoise, irfnoise, ivpvs, iiter = self.sampling.init_lastmodel_fromchain(self.extract_fromlastmodel_path, chainidx)
            self._init_existmodel( imodel,iswdnoise, irfnoise, ivpvs)
            if phase == 1:
                self.iiter += iiter * self.thinning
            else:
                self.iiter = iiter * self.thinning
        else:
         
            self._init_model_and_currentvalues()


        # Set up the modifications 

    
        self.modelmods = ['vsmod', 'vmod']
        self.dimmods = ['birth', 'death']
        self.noisemods = [mod for mod in ['rfnoise', 'swdnoise'] if hasattr(self.sampling, f'{mod}inds') and getattr(self.sampling, f'{mod}inds') is not None and len(getattr(self.sampling, f'{mod}inds')) != 0]
    
        if type(self.priors['vpvs']) == np.float64:
            self.vpvsmods = []
        else:
            self.vpvsmods = ['vpvs']
                  
        self.modifications = self.modelmods + \
            self.dimmods  + self.vpvsmods + self.noisemods

        # Start interate
        pre_itr_time = 0
        while self.iiter < self.iter_phase2:
            #print ("IITER", self.iiter )
            tnull = gettime.time()
            self.iterate()

            if self.iiter % self.thinning ==0:
                self.append_currentmcvalues()
                self.append_currentmodel2()
               
            current_itr_time = gettime.time()-tnull
            if pre_itr_time == 0:
                pre_itr_time = tnull -tnull0
            meantime = (pre_itr_time + current_itr_time) * 0.5
            pre_itr_time = meantime
            

            if self.iiter % 5000 == 0 and  self.iiter != -self.iter_phase1 :
                print(f'chainidx {self.chainidx:.4f}, self.iiter {self.iiter:.4f}, itrtime {meantime:.4f}')

                current_iterations = self.iiter + self.iter_phase1
                
                acceptrate = float(self.n) / current_iterations * 100.
                print(f'{self.chainidx} IITER {self.iiter} {float(self.n)} ACCEPT {acceptrate:.4f}')
            self.iiter += 1

        acceptrate = float(self.n) / current_iterations * 100.

   
