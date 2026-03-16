# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import numpy as np
import copy
from ctypes import *
import Baytomo.rfmini as rfmini
from Baytomo.Models import Model

#degrees2kilometers = 111.19492664455873
class RFminiModRF(object):
    """
    Forward modeling of receiver functions based on SeisPy (Joachim Saul).
    Attributes:
        obsx (numpy.ndarray): Observed x-data (time vector).
        nobs (int): Number of observations.
        ref (str): Reference type ('prf', 'seis', 'srf').

    Methods:
        __init__(self, obsx, nobs, ref, baz=0):
            Initializes the RFraysumModRF object.

        _init_obsparams(self):
            Extracts parameters from observed x-data (time vector).

        init_grids_model(self, grids, staingrididx):
            Initializes the grids model.

        init_arrays(self):
            Initializes arrays for the model.

        write_startmodel(self, h, vp, vsv, vsh, rho, modfile, **params):
            Writes the starting model to a file.

    """
    def __init__(self, obsx, nobs, ref):
        self.ref = ref
        self.obsx = obsx
        self.nobs = nobs
        self._init_obsparams()
        self._init_modelparams()


        self.keys = {'z': '%.2f',
                     'vp': '%.4f',
                     'vs': '%.4f',
                     'rho': '%.4f',
                     'qp': '%.1f',
                     'qs': '%.1f',
                     'n': '%d'}
        self.cacl = True
    def _init_obsparams(self):
        """Extract parameters from observed x-data (time vector).

        fsamp = sampling frequency in Hz
        tshft = time shift by which the RF is shifted to the left
        nsamp = number of samples, must be 2**x
        """

        # get fsamp
        deltas = np.round((self.obsx[1:] - self.obsx[:-1]), 4)
        if np.unique(deltas).size == 1:
            dt = float(deltas[0])
            self.fsamp = 1. / dt
        else:
            raise ValueError("Target: %s. Sampling rate must be constant."
                             % self.ref)
        # get tshft
        self.tshft = -self.obsx[0]

        # get nsamp
        ndata = self.obsx.size
        self.nsamp = 2.**int(np.ceil(np.log2(ndata * 2)))
        
    def _init_modelparams(self):
        """
        Initializes model parameters.
        """
        if self.ref in ['prf', 'seis']:
            self.modelparams = {'wtype': 'P'}
        elif self.ref in ['srf']:
            self.modelparams = {'wtype': 'SV'}
        gauss = np.ones(self.nobs)
        p =  np.ones(self.nobs ) *1.0
        water =  np.ones(self.nobs ) *6.4
        nsv = np.full(self.nobs, None, dtype=object)

        self.modelparams.update(
            {'gauss': gauss,
             'p': p,
             'water': water,
             'nsv': nsv
             })

    def init_grids_model(self, grids, staingrididx):
        """
        Initializes the grids model.

        Parameters:
            grids: The grids object.
            staingrididx: Indices of the nearest grids.

        """
        self.nx = grids.nx
        self.ny = grids.ny
        self.nz = grids.nz
        
        self.staingrididx = staingrididx
        self.dz = grids.dz/grids.scaling 
        self.zmax =  grids.zmax/grids.scaling 
        self.scaling = grids.scaling

        
        self.init_arrays()
    
    def init_arrays(self):
        """
        Initializes arrays for the model.
        """              
        self.current_vp = np.ones((self.nx*self.ny, self.nz))*np.nan
        self.current_vs = np.ones((self.nx*self.ny, self.nz))*np.nan
        self.current_ra = np.ones((self.nx*self.ny, self.nz))*np.nan
        self.current_rho = np.ones((self.nx*self.ny, self.nz))*np.nan      

        self.current_rf = np.ones((self.nobs, len(self.obsx)))*np.nan
        self.prop_rf = np.ones((self.nobs, len(self.obsx)))*np.nan   
        
    def write_startmodel(self, h, vp, vs, rho, modfile, **params):
        """
        Prepares initial model parameters for forward computation.
        """     
        qp = params.get('qp', np.ones(h.size) * 500)
        qs = params.get('qs', np.ones(h.size) * 225)

        z = np.cumsum(h)
        z = np.concatenate(([0], z[:-1]))

        mparams = {'z': z, 'vp': vp, 'vs': vs, 'rho': rho,
                   'qp': qp, 'qs': qs}
        mparams = dict((a, b) for (a, b) in mparams.items()
                       if b is not None)
        pars = mparams.keys()

        nkey = 0
        header = []
        mline = []
        data = np.empty((len(pars), mparams[pars[0]].size))
        for key in ['z', 'vp', 'vs', 'rho', 'qp', 'qs']:
            if key in pars:
                header.append(key)
                mline.append(self.keys[key])
                data[nkey, :] = mparams[key]
                nkey += 1

        header = '\t'.join(header) + '\n'
        mline = '\t'.join(mline) + '\n'

        with open(modfile, 'w') as f:
            f.write(header)
            for i in np.arange(len(data[0])):
                f.write(mline % tuple(data.T[i]))

    def set_modelparams(self, **mparams):
        """
        Update model parameters for forward computation.
        """    
        self.modelparams.update(mparams)

    def compute_rf(self, idx, h, vp, vs, rho, p= 6.4, **params):
        """
        Compute RF using self.modelsparams (dict) for parameters.
        e.g. usage: self.set_modelparams(gauss=1.0)

        Parameters are:
        # idx idx of rf position
        # z  depths of the top of each layer
        gauss: Gauss parameter
        water: water level
        p: angular slowness in sec/deg
        wtype: type of incident wave; must be 'P' or 'SV'
        nsv: tuple with near-surface S velocity and Poisson's ratio
            (will be computed by input model, if None)
        """
        gauss = self.modelparams['gauss']#[idx]
        water = self.modelparams['water']#[idx]
        #p = p#self.modelparams['p']#/degrees2kilometers
        p = float(self.modelparams['p'][idx]) #laptop
        wtype = self.modelparams['wtype']#[idx]
        nsv = self.modelparams['nsv'][idx]
        
        qp = params.get('qp', np.ones(h.size) * 500.)
        qs = params.get('qs', np.ones(h.size) * 225.)

        z = np.cumsum(h)
        z = np.concatenate(([0], z[:-1]))

        nsvp, nsvs = float(vp[0]), float(vs[0])
        vpvs = nsvp / nsvs
        poisson = (2 - vpvs**2)/(2 - 2 * vpvs**2)

        if nsv is None:
            nsv = nsvs

        time = np.arange(self.nsamp) / self.fsamp - self.tshft
        fp, fsv, qrf = rfmini.synrf(
            z, vp, vs, rho, qp, qs,
            p, gauss, self.nsamp, self.fsamp,
            self.tshft, nsv, poisson, wtype)

        # must be converted to float64
        qrfdata = qrf.astype(float)

        return time[:self.obsx.size], qrfdata[:self.obsx.size]

    def check_diff(self, vp, vs, ra, rho):
        """
        Check for differences between the current model and the proposed model.
        The method calculates the differences based on the sum of absolute differences in each row for the specified parameters. 
        If the sum is not equal to zero, it identifies the indices where differences exist and returns a unique set of these indices. 
        These indices represent the (x, y) profiles that should be recalculated.

        Parameters:
            vp (numpy.ndarray): Array of P-wave velocities for the proposed model.
            vs (numpy.ndarray): Array of S-wave velocities for the proposed model.
            ra (numpy.ndarray): Array of density values for the proposed model.
            rho (numpy.ndarray): Array of density values for the proposed model.
            von (numpy.ndarray): Array of orientations for the proposed model.

        Returns:
            numpy.ndarray: Array of unique indices where differences are found.
        """
        def cal_diff(prop_param, current_param):
            diff = prop_param-current_param
            sum_ = np.sum(diff, axis=1)
            diffidx = np.where(sum_ != 0)[0]
            return diffidx

        diffidxs = []
        prop_params = [vp, vs, ra, rho]
        current_params = [self.current_vp, self.current_vs,
                          self.current_ra, self.current_rho]
        
        for i, prop_param in enumerate(prop_params):
            diffidx = cal_diff(prop_param, current_params[i])
            diffidxs.append(diffidx)
        diffidxs = np.concatenate(diffidxs)
        return np.unique(diffidxs)

    def accept_as_currentmoddata(self):
        self.current_rf  = copy.deepcopy(self.prop_rf)
        self.current_vp = copy.deepcopy(self.prop_vp)
        self.current_vs = copy.deepcopy(self.prop_vs)
        self.current_ra = copy.deepcopy(self.prop_ra)
        self.current_rho = copy.deepcopy(self.prop_rho)
        
    def run_model(self,  vp, vs, ra, rho, **params):
        self.cacl = False
        # check if smallest velocity on the top layer
        #prop_vs = vs.reshape(self.nx * self.ny, self.nz)
        #smallest_in_row = np.min(prop_vs, axis=1)  # Find the smallest value in each row
        #mask = prop_vs[:, 0] == smallest_in_row  # Create a mask for rows where the first element is the smallest
        #result = np.all(mask)  # Check if all rows satisfy the condition

         # reshape the model into vs, vp, ra - depth structure
        self.prop_vp = vp.reshape(self.nx*self.ny, self.nz)
        self.prop_vs = vs.reshape(self.nx*self.ny, self.nz)
        self.prop_ra = ra.reshape(self.nx*self.ny, self.nz)
        self.prop_rho = rho.reshape(self.nx*self.ny, self.nz)
        
        #if any(not np.all(min(row) == row[0]) for row in self.prop_vs):
        if not np.all(self.prop_vs.min(axis=1) == self.prop_vs[:, 0]):
            return np.nan, np.nan
        difflocs = self.check_diff(self.prop_vp, self.prop_vs, self.prop_ra, self.prop_rho)
        
        if len(difflocs) == 0:
            return self.current_time, self.current_rf
        self.prop_rf  = copy.deepcopy(self.current_rf)
         
        for i in range(self.nobs):
            loc = self.staingrididx[i]
            if loc not in difflocs: #If i in diffidxs then do syn
                continue

            sta_vp = self.prop_vp[loc,:]
            sta_vs = self.prop_vs[loc,:]
            sta_ra = self.prop_ra[loc,:]
            sta_rho = self.prop_rho[loc,:]

	    
            h, sta_vp, sta_vs, sta_ra, sta_rho = Model.get_stepmodel_from_grids(
                self.dz, sta_vp, sta_vs, sta_ra, sta_rho, self.zmax)
          


            h = h.astype(float)
            sta_vp = sta_vp.astype(float)
            sta_vs = sta_vs.astype(float)
            sta_rho = sta_rho.astype(float)
            

                
            time, qrf = self.compute_rf(i, h, sta_vp, sta_vs, sta_rho, **params)
            self.current_time = time
            self.prop_rf[i] = qrf
        self.cacl = True       
        return self.current_time, self.prop_rf
