#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:40:01 2021

@author: kuanyu
"""
import ctypes
import sys
from ctypes import *
from Baytomo.surfdisp96_ext import surfdisp96
from Baytomo.time_2d  import time_2d_wrapper  
from Baytomo.Models import Model
import numpy as np
import copy
import time
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
import multiprocessing as mp
from collections import OrderedDict 


class FMST(object):
    """
        Parameters
        ----------
        vp : TYPE
            DESCRIPTION.
        vs : TYPE
            DESCRIPTION.
        ra : TYPE
            DESCRIPTION.
        rho : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
    """

    def __init__(self, grids):

        # grid info
        self.nx = grids.nx
        self.ny = grids.ny
        self.dx = grids.dx #* grids.scaling 
        self.dy = grids.dy #* grids.scaling
        
        self.xmin = grids['gridx'][0]
        self.ymin = grids['gridy'][0]

        self.k = 0
        self.eps = 0.001
        self.messages = 0

    def bilin(self, a, b,  c, d):
        if (c-d) == 0.:
            return (0)

        return ((a-b)/(c-d))

    def ttime(self, x, y, tbuf, nz, printout =False):

        ix = int(np.floor(x))
        iy = int(np.floor(y))

        x1 = ix
        y1 = iy
        x2 = ix+1
        y2 = iy+1

        i = iy + (ix * nz)
        v1 = tbuf[int(i)]

        i = iy + 1 + (ix * nz)
        v2 = tbuf[int(i)]

        i = iy + ((ix+1) * nz)
        v3 = tbuf[int(i)]

        i = iy + 1 + ((ix+1) * nz)
        v4 = tbuf[int(i)]

        a = self.bilin(y2, y, y2, y1)
        b = self.bilin(x2, x, x2, x1)
        c = self.bilin(x, x1, x2, x1)
        d = self.bilin(y, y1, y2, y1)
        e = self.bilin(x2, x, x2, x1)
        f = self.bilin(x, x1, x2, x1)

        v = (a * b * v1) + (a * c * v4) + (d * e * v2) + (d * f * v3)

        return v

    def calc_tt(self, vmod, xs, ys):
        tbuf = np.zeros((self.nx*self.ny), dtype='float32')
        hsbuf = np.ones((self.nx*self.ny), dtype='float32')*np.nan
        k = 0
        for i in range(self.nx):
            for j in range(self.ny):
                hsbuf[k] = self.dx/vmod[i][j]

                k += 1

        xst = (xs -self.xmin)/self.dx
        yst = (ys -self.ymin)/self.dy
        
        # numpy covert to ctypes data
        #hsbuf = (c_float * len(hsbuf))(*hsbuf)
        #tbuf = (c_float * len(tbuf))(*tbuf)
        #             c_float(yst), c_float(self.eps), c_int(self.messages))
        #time_2d(hsbuf, tbuf, c_int(self.nx), c_int(self.ny), c_float(xst),
        #             c_float(yst), c_float(self.eps), c_int(self.messages))

        # ctypes covert to numpy
        #floatPtr = ctypes.cast(tbuf, ctypes.POINTER(ctypes.c_float))
        #floatList = [floatPtr[i] for i in range(self.nx*self.ny)]
        #tbuf = np.array(floatList)
        tbuf = time_2d_wrapper(hsbuf, tbuf, self.nx, self.ny, xst,yst, self.eps,self.messages)      
        return tbuf



class SurfDisp(object):
    """Forward modeling of dispersion curves based on surf96 (Rob Herrmann).
    The quick fortran routine is from Hongjian Fang:
        https://github.com/caiweicaiwei/SurfTomo
    BayHunter.SurfDisp leaning on the python wrapper of Marius Isken:
        https://github.com/miili/pysurf96
    """

    def __init__(self, obsx, ref):
        self.obsx = obsx
        self.kmax = obsx.size
        self.ref = ref

        self.modelparams = {
            'mode': 1,  # mode, 1 fundamental, 2 first higher
            'flsph': 0  # flat earth model
        }

        self.wavetype, self.veltype = self.get_surftags(ref)

        if self.kmax > 60:
            message = "Your observed data vector exceeds the maximum of 60 \
periods that is allowed in SurfDisp. For forward modeling SurfDisp will \
reduce the samples to 60 by linear interpolation within the given period \
span.\nFrom this data, the dispersion velocities to your observed periods \
will be determined. The precision of the data will depend on the distribution \
of your samples and the complexity of the input velocity-depth model."
            self.obsx_int = np.linspace(obsx.min(), obsx.max(), 60)
            print(message)

    def set_modelparams(self, **mparams):
        self.modelparams.update(mparams)

    def get_surftags(self, ref):
        if ref == 'rdispgr':
            return (2, 1)

        elif ref == 'ldispgr':
            return (1, 1)

        elif ref == 'rdispph':
            return (2, 0)

        elif ref == 'ldispph':
            return (1, 0)
        else:
            tagerror = "Reference is not available in SurfDisp. If you defined \
a user Target, assign the correct reference (target.ref) or update the \
forward modeling plugin with target.update_plugin(MyForwardClass()).\n \
* Your ref was: %s\nAvailable refs are: rdispgr, ldispgr, rdispph, ldispph\n \
(r=rayleigh, l=love, gr=group, ph=phase)" % ref
            raise ReferenceError(tagerror)

    def get_modelvectors(self, h, vp, vs, rho):

        nlayer = len(h)
        thkm = np.zeros(100)
        thkm[:nlayer] = h

        vpm = np.zeros(100)
        vpm[:nlayer] = vp

        vsm = np.zeros(100)
        vsm[:nlayer] = vs

        rhom = np.zeros(100)
        rhom[:nlayer] = rho

        return thkm, vpm, vsm, rhom

    def run_model(self, h, vp, vs, ra, rho, methods=3, **params):
        """ The forward model will be run with the parameters below.
        thkm, vpm, vsm, rhom: model for dispersion calculation
        nlayer - I4: number of layers in the model
        iflsph - I4: 0 flat earth model, 1 spherical earth model
        iwave - I4: 1 Love wave, 2 Rayleigh wave
        mode - I4: ith mode of surface wave, 1 fundamental, 2 first higher, ...
        igr - I4: 0 phase velocity, > 0 group velocity
        kmax - I4: number of periods (t) for dispersion calculation
        t - period vector (t(NP))
        cg - output phase or group velocities (vector,cg(NP))
        """
        nlayer = len(h)

        iflsph = self.modelparams['flsph']
        mode = self.modelparams['mode']
        iwave = self.wavetype
        igr = self.veltype

        vs = Model.get_vsv_vsh(vs, ra, vs_type=iwave, methods=methods)
        h, vp, vs, rho = self.get_modelvectors(h, vp, vs, rho)

        if self.kmax > 60:
            kmax = 60
            pers = self.obsx_int

        else:
            pers = np.zeros(60)
            kmax = self.kmax
            pers[:kmax] = self.obsx

        dispvel = np.zeros(60)  # result
        error = surfdisp96(h, vp, vs, rho, nlayer, iflsph, iwave,
                           mode, igr, kmax, pers, dispvel)
        if error == 0:
            if self.kmax > 60:
                disp_int = np.interp(self.obsx, pers, dispvel)
                return self.obsx, disp_int

            return pers[:kmax], dispvel[:kmax]

        return np.nan, np.nan


class surf_forward(object):
    """ 2 step forward modelling:
    1st step : local dispersion curves at each grid point
    2nd step: extract every period from the 1st step and
    """

    def __init__(self, obsx, stas, pairs, pairsinprds, ref, n_jobs=1):
        self.stas = stas
        self.pairs = pairs
        self.srcs =  self.pairs[:, 0]
        self.rcvs = self.pairs[:, 1]
        self.pairsinprds = pairsinprds
        self.surfdisp = SurfDisp(obsx, ref)
        self.periods = obsx
        self.cacl = False
        self.n_jobs = n_jobs

    def init_grids_model(self, grids, level=0):
        self.nx = grids.nx
        self.ny = grids.ny
        self.nz = grids.nz

        self.dx = grids.dx
        self.dy = grids.dy
        self.dz = grids.dz
        
        self.xmin = grids['gridx'][0]
        self.ymin = grids['gridy'][0]
        
        self.zmax = grids.zmax/grids.scaling
        self.scaling = grids.scaling

        self.init_1D_swd()
        self.init_2D_swd()
        
        self.fmst = FMST(grids)

    def init_1D_swd(self):
        self.current_vp = np.ones((self.nx*self.ny, self.nz))*np.nan
        self.current_vs = np.ones((self.nx*self.ny, self.nz))*np.nan
        self.current_ra = np.ones((self.nx*self.ny, self.nz))*np.nan
        self.current_rho = np.ones((self.nx*self.ny, self.nz))*np.nan

        self.current_surfdisp = np.ones(
            (self.nx*self.ny, len(self.periods)))*np.nan

    def init_2D_swd(self):
        self.current_ttm = np.ones((len(self.pairs), len(self.periods)))*np.nan

    def check_diff(self, vp, vs, ra, rho):
        """
        1. Minus two matrix
        2. sum each row => oder in (x,y)
        3. If !=0, then different, that (x,y) profile shoud be calculated
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
        self.current_surfdisp  = copy.deepcopy(self.prop_surfdisp)
        self.current_vp = copy.deepcopy(self.prop_vp)
        self.current_vs = copy.deepcopy(self.prop_vs)
        self.current_ra = copy.deepcopy(self.prop_ra)
        self.current_rho = copy.deepcopy(self.prop_rho)
        self.current_ttm = copy.deepcopy(self.prop_ttm)

    def _moddata_valid(self, x, y):
        if not type(x) == np.ndarray:
            return False
        if not len(self.periods) == len(x):
            return False
        if not np.sum(self.periods - x) <= 1e-5:
            return False
        if not len(self.periods) == len(y):
            return False

        return True

    def _valid_model(self,h, ):
        # check whether nlayers lies within the prior
        layermin = 0#
        layermax = 20
        layermodel = (h.size - 1)
        thickmin = 1

        if not (layermodel >= layermin and layermodel <= layermax):
            return False

        # check model for layers with thicknesses of smaller thickmin
        if np.any(h[:-1] < thickmin):
            return False
        return True

    def run_surf96(self, i):
        i = int(i)
        h, vp_i, vs_i, ra_i, rho_i = Model.get_stepmodel_from_grids(
                    self.dz, self.prop_vp[i], self.prop_vs[i], self.prop_ra[i], self.prop_rho[i], self.zmax)
        h = h/self.scaling
                                
        if not self._valid_model(h):
            return np.nan

        x, y = self.surfdisp.run_model(h, vp_i, vs_i, ra_i, rho_i)
     

        if not self._moddata_valid( x, y) or  y is np.nan:
            
            return np.nan
        else:
            return y

        
        
    def run_surf96_par(self, j, n_in_jobs,difflocs):
        current_surfdisp = np.ones((n_in_jobs, len(self.periods)))*np.nan
        tnull =time.time()
        for k,i in enumerate(difflocs[j]):
            i = int(i)
            if i == -1:
                continue
            
            h, vp_i, vs_i, ra_i, rho_i = Model.get_stepmodel_from_grids(
                    self.dz, self.prop_vp[i], self.prop_vs[i], self.prop_ra[i], self.prop_rho[i], self.zmax)
            h = h/self.scaling



            if not self._valid_model(h):
                return current_surfdisp

            x, y = self.surfdisp.run_model(h, vp_i, vs_i, ra_i, rho_i)
            

            if not self._moddata_valid( x, y):
                return current_surfdisp
            else:
                current_surfdisp[k] = y
        current_surfdisp = current_surfdisp[~np.isnan(current_surfdisp).any(axis=1)]
        return current_surfdisp
    


    def run_ttm(self):
        current_ttm = np.ones((len(self.pairs), len(self.periods)))*np.nan
        
        for i, prd in enumerate(self.periods):
            vmod = self.prop_surfdisp[:, i].reshape(self.nx, self.ny)

            "if the array is the same as before, do no recalculate again"
            vmod_current = self.current_surfdisp[:, i].reshape(self.nx, self.ny)
            if np.array_equal(vmod, vmod_current):
                current_ttm[:, i] = copy.deepcopy(self.current_ttm[:, i])
                continue
            
            tt = np.ones(len(self.pairs)) * np.nan
            idx = -1
            for j, srcidx in enumerate(self.srcs):

                if idx != srcidx:
                    idx = srcidx
                    xs, ys = self.stas[idx]
                    ttmfield = self.fmst.calc_tt(vmod, xs, ys)
                # not every pair has ttm for every period
                if j not in self.pairsinprds[prd]:
                    tt[j] = -1
                    continue
                
                rcvidx = self.rcvs[j]
                xrt = (self.stas[rcvidx][0]-self.xmin)/self.dx

                yrt = (self.stas[rcvidx][1]-self.ymin)/self.dy

                tt[j] = self.fmst.ttime(xrt, yrt, ttmfield, self.ny)
            current_ttm[:, i] = tt

        return current_ttm



    def run_model(self, vp=None, vs=None, ra=None, rho=None, **kwargs):
        """ 2 step forward modelling:
     1st step : local dispersion curves at each grid point, store every curve
     at each grid point until next time. Do a comparison to see which curves should
     be changed
     2nd step: extract every period from the 1st step
     """
        self.cacl = False


        # reshape the model into vs, vp, ra - depth structure
        self.prop_vp = vp.reshape(self.nx*self.ny, self.nz)
        self.prop_vs = vs.reshape(self.nx*self.ny, self.nz)
        self.prop_ra = ra.reshape(self.nx*self.ny, self.nz)
        self.prop_rho = rho.reshape(self.nx*self.ny, self.nz)

        # check if smallest velocity on the top layer
        if not np.all(self.prop_vs.min(axis=1) == self.prop_vs[:, 0]):
            return np.nan, np.nan


        difflocs = self.check_diff(self.prop_vp, self.prop_vs, self.prop_ra, self.prop_rho)
        
        if len(difflocs) == 0:
            return self.periods, self.current_ttm


        self.prop_surfdisp  = copy.deepcopy(self.current_surfdisp)

        # run 1D forward modelling
        if len(difflocs) >= 600 and self.n_jobs >1:
            
            difflocs_ = np.concatenate((difflocs,np.ones(self.n_jobs-len(difflocs)%self.n_jobs)*(-1)))
            
            n_in_jobs = int(len(difflocs_)/self.n_jobs)
            difflocs_ = difflocs_.reshape(self.n_jobs,n_in_jobs)
            prop_surfdisp = Parallel(n_jobs=self.n_jobs, backend= 'multiprocessing' )(delayed(self.run_surf96_par)( j, n_in_jobs,difflocs_) for j in range(self.n_jobs))
        
           
            prop_surfdisp = np.concatenate((prop_surfdisp))
            if np.isnan(prop_surfdisp).any():
                return np.nan, np.nan

            else:
                self.prop_surfdisp[difflocs] = prop_surfdisp

        else:
            for i in difflocs:
                y = self.run_surf96( i)
                if type(y) is float and y is np.nan:
                    return np.nan, np.nan
                else:
                    self.prop_surfdisp[i] = y
                    
        # run 2D forward modelling
        # The c code is recursive so it's impossbile to run them in parallel
        self.prop_ttm =self.run_ttm()


        self.cacl = True
        return self.periods, self.prop_ttm
