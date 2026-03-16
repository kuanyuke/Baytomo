#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:46:56 2022

@author: kuanyu
"""

import numpy as np
import copy
from numpy.fft import fft, ifft
from scipy.fftpack import hilbert, fft, ifft, next_fast_len

try:
    from scipy.signal import sosfilt
    from scipy.signal import zpk2sos
except ImportError:
    from ._sosfilt import _sosfilt as sosfilt
    from ._sosfilt import _zpk2sos as zpk2sos
from Baytomo.Models import Model
import sys
import Baytomo.raysum as raysum
from ctypes import *
import gc
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
import random
import time as timer 
degrees2kilometers = 111.19492664455873
log_file = "raysum_timeout_log.txt"
import signal
import time

# put near top of raysum.py
import multiprocessing as mp

def _raysum_worker(q, args):
    try:
        res = raysum.raysum_interface(*args)
        q.put(("ok", res))
    except Exception as e:
        # pass back the exception string so parent can raise/log
        q.put(("err", f"{type(e).__name__}: {e}"))

import multiprocessing
import queue
import time
import ctypes
import threading

def terminate_thread(thread):
    """Force kill a thread - works for C/Fortran code"""
    if not thread.is_alive():
        return
        
    # Get thread ID
    tid = ctypes.c_long(thread.ident)
    
    # Send exception to the thread
    exception = ctypes.py_object(SystemExit)
    count = ctypes.c_ulong(1)
    
    # This forces the thread to exit
    ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, exception)
    
    # Wait a bit
    thread.join(timeout=0.1)
    
    # If still alive, use more aggressive method
    if thread.is_alive():
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)

def run_raysum_thread_timeout(args, timeout=10):
    """
    Thread-based timeout with forced termination
    """
    result_container = []
    error_container = []
    
    def raysum_worker():
        try:
            result = raysum.raysum_interface(*args)
            result_container.append(result)
        except Exception as e:
            error_container.append(str(e))
    
    thread = threading.Thread(target=raysum_worker)
    thread.daemon = True  # Thread will die if main thread exits
    thread.start()
    
    start_time = time.time()
    thread.join(timeout)
    
    if thread.is_alive():
        # Force kill the thread
        print(f"KILLING raysum thread after {timeout}s")
        terminate_thread(thread)
        return None, "TIMEOUT_KILLED"
    
    # Check results
    if result_container:
        return result_container[0], None
    elif error_container:
        return None, error_container[0]
    else:
        return None, "UNKNOWN_ERROR"


# Signal handler to raise an exception when time limit is exceeded
##def handler(signum, frame):
#    raise TimeoutError("Timeout reached")

#def run_raysum_interface(*args):
#   try:
#        result = raysum.raysum_interface(*args)
#        return result
#    except Exception as e:
#        return None


##def _pad(array, n):
#    """
#    Pad an array with zeros up to a specified length.

 #   Parameters:
#        array (numpy.ndarray): Input array.
 #       n (int): Target length.

 #   Returns:
 #       numpy.ndarray: Padded array.
 #   """    
 ##   tmp = np.zeros(n)
 #   tmp[:array.shape[0]] = array
 #   return tmp

def _gauss_filt(dt, nft, f0):
    """
    Generate a Gaussian filter for signal processing.

    Parameters:
        dt (float): Time step.
        nft (int): Number of points in the frequency domain.
        f0 (float): Cutoff frequency.

    Returns:
        numpy.ndarray: Gaussian filter.
    """
    df = 1./(nft*dt)
    nft21 = int(0.5*nft + 1)
    f = df*np.arange(nft21)
    w = 2.*np.pi*f
    gauss = np.zeros(nft)
    gauss[:nft21] = np.exp(-0.25*(w/f0)**2.)/dt
    gauss[nft21:] = np.flip(gauss[1:nft21-1])
    return gauss
    
def deconv_waterlevel(i, trROT, npts, fsamp, dt, waterlevel=0.001, gfilt=1, rot=2):
    """
    Deconvolve the water level from a signal.

    Parameters:
        i (int): Index.
        trROT (numpy.ndarray): 3D array representing rotated seismograms.
        npts (int): Number of points.
        fsamp (float): Sampling frequency.
        dt (float): Time step.
        waterlevel (float): Water level for deconvolution. Default is 0.001.
        gfilt (float): Gaussian filter parameter. Default is 1.
        rot (int): Rotation type (1 or 2). Default is 2.

    Returns:
        list: List of deconvolved signals.
    """
    if rot == 2:
        src = trROT[0,:,i].flatten().copy()
        rtr = trROT[1,:,i].flatten().copy()
    elif rot == 1:
        src = trROT[2,:,i].flatten().copy()
        rtr = trROT[0,:,i].flatten().copy()
    
    if np.any(np.isnan(src)):
        return np.nan
        
    dts = len(src)*dt/2.
    nn = int(round((dts-5.)*fsamp)) + 1
            
    npad = int(npts)
    freqs = np.fft.fftfreq(npad, d=dt)

    # Fourier transform
    Fp = np.fft.fft(src, n=npad)
    Fd1 = np.fft.fft(rtr, n=npad)

    # Auto and cross spectra
    Spp = np.real(Fp*np.conjugate(Fp))
    Sd1p = Fd1*np.conjugate(Fp)

    # Final processing depends on method
    phi = np.amax(Spp)*waterlevel
    Sdenom = Spp
    Sdenom[Sdenom < phi] = phi
    
    if all(v == 0 for v in src) or all(v == 0 for v in Sdenom):
        return np.nan
    
    # Apply Gaussian filter?
    if gfilt:
    	gauss = _gauss_filt(dt, npad, gfilt)
    	gnorm = np.sum(gauss)*(freqs[1]-freqs[0])*dt
    else:
    	gauss = np.ones(npad)
    	gnorm = 1.
    

    rf_list = []
    
    rfd1 = np.fft.ifftshift(np.real(np.fft.ifft(
                gauss*Sd1p/Sdenom))/gnorm)
    rf_list.append(rfd1)
                
    return rf_list                  


class RFraysumModRF(object):
    """
    Forward modeling of receiver functions based on Raysum (Andrew Frederiksen).

    Attributes:
        obsx (numpy.ndarray): Observed x-data (time vector).
        nobs (int): Number of observations.
        ref (str): Reference type ('prf', 'seis', 'srf').
        baz (float): Back-azimuth. Default is 0.

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

    def __init__(self, obsx, nobs, ref, baz=0):
        """
        Initializes the RFraysumModRF object.

        Parameters:
            obsx (numpy.ndarray): Observed x-data (time vector).
            nobs (int): Number of observations.
            ref (str): Reference type ('prf', 'seis', 'srf').
            baz (float): Back-azimuth. Default is 0.
        """
        self.ref = ref
        self.obsx = obsx
        self.nobs = nobs
        self.obsbaz = baz
        self._init_obsparams()
        self._init_modelparams()
        if self.ref in ['prf', 'seis']:
            self.modelparams = {'wtype': 'P'}
        elif self.ref in ['srf']:
            self.modelparams = {'wtype': 'SV'}


        self.keys = {'z': '%.2f',
                     'vp': '%.4f',
                     'vs': '%.4f',
                     'rho': '%.4f',
                     'qp': '%.1f',
                     'qs': '%.1f',
                     'n': '%d'}
        self.maxlayer =15
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
        """Initializes model parameters.
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

        
        self.init_arrays()
    
    def init_arrays(self):
        """
        Initializes arrays for the model.
        """        
        self.current_vp = np.ones((self.nx*self.ny, self.nz))*np.nan
        self.current_vs = np.ones((self.nx*self.ny, self.nz))*np.nan
        self.current_ra = np.ones((self.nx*self.ny, self.nz))*np.nan
        self.current_rho = np.ones((self.nx*self.ny, self.nz))*np.nan      
        self.current_von = np.ones((self.nx*self.ny, self.nz, 3))*np.nan  
        self.current_rf = np.ones((self.nobs, len(self.obsx)))*np.nan
        self.prop_rf = np.ones((self.nobs, len(self.obsx)))*np.nan   

    def write_startmodel(self, h, vp, vsv, vsh, rho, modfile, **params):
        qp = params.get('qp', np.ones(h.size) * 500)
        qs = params.get('qs', np.ones(h.size) * 225)

        z = np.cumsum(h)
        z = np.concatenate(([0], z[:-1]))

        mparams = {'z': z, 'vp': vp, 'vsv': vsv, 'vsh': vsh, 'rho': rho,
                   'qp': qp, 'qs': qs}
        mparams = dict((a, b) for (a, b) in mparams.items()
                       if b is not None)
        pars = mparams.keys()

        nkey = 0
        header = []
        mline = []
        data = np.empty((len(pars), mparams[pars[0]].size))
        for key in ['z', 'vp', 'vsv', 'vsh', 'rho', 'qp', 'qs']:
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
        self.modelparams.update(mparams)

    def create_ani_value(self,nlyar, ani):
        iso = np.ones(nlyar)
        plunge = np.zeros(nlyar)

        for i, ra in enumerate(ani):
            if ra != 0:
                iso[i] = 0
                plunge[i] = 90
        return iso, plunge

    def buildmodel(self, h, vp, vs, ra, den):
        """
        Build a input model for Raysum.

        Parameters:
            h (numpy.ndarray): Array of layer thicknesses.
            vp (numpy.ndarray): Array of P-wave velocities.
            vs (numpy.ndarray): Array of S-wave velocities.
            ra (numpy.ndarray): Array of density values.
            den (float): Density value for the entire model.

        Returns:
            tuple: Tuple containing model parameters (nlyar, thick, alpha, beta, iso, ds, rho, plunge).
        """

        nlyar = len(h)
        
        niso, nplunge = self.create_ani_value(nlyar, ra)

        thick = np.zeros(self.maxlayer)
        rho = np.zeros(self.maxlayer)
        alpha = np.zeros(self.maxlayer)
        beta = np.zeros(self.maxlayer)

        # ani
        iso = np.zeros(self.maxlayer)
        ds = np.zeros(self.maxlayer)
        plunge = np.zeros(self.maxlayer)

        thick[:nlyar] = h * 1000
        rho[:nlyar] = den * 1000
        alpha[:nlyar] = vp * 1000
        beta[:nlyar] = vs * 1000

        # ani parameter
        ani = (np.asarray(ra) + 100.) * 0.01
        ds[:nlyar] = ani
        iso[:nlyar] = niso
        plunge[:nlyar] = nplunge

        return nlyar, thick, alpha, beta, iso, ds, rho, plunge

        
    def compute_rf(self, idx, h, vp, vs, ra, rho, strikee, dipp, bazz,p= 6.4,**params):
        """
        Compute RF using self.modelsparams (dict) for parameters.
        e.g. usage: self.set_modelparams(gauss=1.0)

        Parameters are:
        # z  depths of the top of each layer
        gauss: Gauss parameter
        water: water level
        p: angular slowness in sec/deg
        wtype: type of incident wave; must be 'P' or 'SV'
        nsv: tuple with near-surface S velocity and Poisson's ratio
            (will be computed by input model, if None)
        """

        gauss = self.modelparams['gauss'][idx]
        water = self.modelparams['water']
        p = float(self.modelparams['p'][idx])
        p = p/degrees2kilometers
        wtype = self.modelparams['wtype']

        if wtype == 'P':
            iphase = 1  # initial phase index (1 - P; 2 - SV; 3 - SH)
        elif wtype == 'SV':
            iphase = 2

        mults = 2  # Multiples: 0 for none, 1 for Moho, 2 for all first-order
        align = 1  # Alignment: 0 is none, 1 aligns on primary phase (P or S)
        shift = self.tshft  # Shift of traces -- t=0 at this time (sec)
        out_rot = 2  # Rotation to output: 0 is NS/EW/Z, 1 is R/T/Z, 2 is P/SV/SH

        phname_in = ''

        # Geometry
        baz = np.zeros(200)
        slow = np.zeros(200)
        sta_dx = np.zeros(200)
        sta_dy = np.zeros(200)
        ntr = len(bazz)
        slow[:ntr] = p * 0.001
        baz[:ntr] = bazz

        #Model 
        dp = np.zeros(self.maxlayer)
        trend = np.zeros(self.maxlayer)
        strike = np.zeros(self.maxlayer)
        dip = np.zeros(self.maxlayer)
        
        
        nlyar = len(h)
        dip[:nlyar] = dipp
        strike[:nlyar] = strikee

        time = np.arange(self.nsamp) / self.fsamp - self.tshft
        nsamp = self.nsamp * 2
        dt = 1./self.fsamp

        
        nlyar, h, vp, vs, iso, ds, rho, plunge = self.buildmodel( h, vp, vs, ra, rho)

        args = (nlyar, h, rho, vp, vs,
                dp, ds, trend,
                plunge, strike, dip, iso, iphase,
                ntr, baz, slow, sta_dx, sta_dy,
                mults, nsamp, dt, gauss, align, self.tshft, out_rot, phname_in)

###        signal.signal(signal.SIGALRM, handler)
###        signal.alarm(10)  # Set timeout in seconds
    
###        try:
###            # Run the function
###           result = raysum.raysum_interface(nlyar, h, rho, vp, vs, dp, ds, trend, plunge, strike, dip, iso, iphase,
###                                             ntr, baz, slow, sta_dx, sta_dy, mults, nsamp, dt, gauss, align, self.tshft, out_rot, phname_in)
###         signal.alarm(0)  # Cancel the alarm once function finishes before timeout
            
###        except TimeoutError:
###            print(f"Timeout reached, stopping computation.")
###            print (nlyar, h, rho, vp, vs, dp, ds, trend, plunge, strike, dip)
###            return np.nan, np.nan  # Or any other appropriate value for timeout cases
###        except Exception as e:
###            print(f"Error in raysum_interface: {e}") 
###            return np.nan, np.nan

#        _,  _, _,  tr_cart, tr_ph = raysum.raysum_interface(nlyar, h, rho, vp, vs,
#                                                   dp, ds, trend,
#                                                   plunge, strike, dip, iso, iphase,
#                                                   ntr, baz, slow, sta_dx, sta_dy,
#                                                   mults, nsamp, dt, gauss, align, self.tshft, out_rot, phname_in)

        result, error = run_raysum_thread_timeout(args, timeout=10)


        if error == "TIMEOUT_KILLED":
            print (f"Raysum timeout/error for station: {error}")
            #print (h, rho, vp, vs,strike, dip)
            return np.nan, np.nan
        if error:
            print (f"Raysum timeout/error for station: {error}")
            #print (h, rho, vp, vs,strike, dip)
            return np.nan, np.nan
    
        _,  _, _,  tr_cart, tr_ph = result
        tshift = int(self.nsamp - int(self.tshft/dt))
        qrfdata = []
        trROTs = tr_ph[:, :int(nsamp), :len(bazz)]


        
        for i in range(len(bazz)):
            RFS =deconv_waterlevel(i, trROTs, nsamp, self.fsamp, dt, waterlevel=0.001, gfilt=1, rot = out_rot)
            if RFS is np.nan:
                #print ("no values")
                return np.nan, np.nan
            RFR = RFS[0]
            qrfdata.append( RFR.astype(float)[tshift:tshift+self.obsx.size])
        
        
        return time[:self.obsx.size], qrfdata

  
    def run_raysum_single(self, i, loc,  **params):
        """
        Run raysum calculation for a single station.

        Parameters:
            i (int): Index of the station.
            loc (int): Location index for the station.
            params (dict): Additional parameters for the raysum calculation.

        Returns:
            tuple: Tuple containing the time vector and computed Receiver Functions.
        """
        sta_vp = self.prop_vp[loc,:]
        sta_vs = self.prop_vs[loc,:]
        sta_ra = self.prop_ra[loc,:]
        sta_rho = self.prop_rho[loc,:]
        sta_von = self.prop_von[loc,:,:]

        h, sta_vp, sta_vs, sta_ra, sta_rho, dip, strike = Model.get_stepmodel_from_grids_dip(
                sta_von, self.dz, sta_vp, sta_vs, sta_ra, sta_rho, self.zmax)

        h = h.astype(float)
        sta_vp = sta_vp.astype(float)
        sta_vs = sta_vs.astype(float)
        sta_ra = sta_ra.astype(float)
        sta_rho = sta_rho.astype(float)
        dip = dip.astype(float)
        strike = strike.astype(float)
        if type(self.obsbaz) in [int, float, np.float64]:
            baz = self.obsbaz
        else:
            baz = self.obsbaz[i]

        
        time, qrf = self.compute_rf(i,h, sta_vp, sta_vs, sta_ra, sta_rho, strike, dip, baz, **params)
        return time, qrf 

    def check_diff(self, vp, vs, ra, rho, von):
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
        prop_params = [vp, vs, ra, rho, von[:, :,0], von[:, :,1], von[:, :,2] ]
        
        current_params = [self.current_vp, self.current_vs,
                          self.current_ra, self.current_rho,
                          self.current_von[:, :,0], self.current_von[:, :,1], self.current_von[:, :,2] ]
        
        for i, prop_param in enumerate(prop_params):
            diffidx = cal_diff(prop_param, current_params[i])
            diffidxs.append(diffidx)
        diffidxs = np.concatenate(diffidxs)
        return np.unique(diffidxs)

    def accept_as_currentmoddata(self):
        """Accept propose model as current model """
        self.current_rf  = copy.deepcopy(self.prop_rf)
        self.current_vp = copy.deepcopy(self.prop_vp)
        self.current_vs = copy.deepcopy(self.prop_vs)
        self.current_ra = copy.deepcopy(self.prop_ra)
        self.current_rho = copy.deepcopy(self.prop_rho)
        self.current_von = copy.deepcopy(self.prop_von)
        


    def run_model(self,   vp, vs, ra, rho,  von,  **params):
        # 'cacl' will later indicate whether the proposed model will be accepted or not.
        # If the modification involves noise parameters, the proposed model remains the same as the current one,
        # and the model is returned; thus, 'cacl' is set to False.
        self.cacl = False

        # check if smallest velocity on the top layer
        prop_vs = vs.reshape(self.nx * self.ny, self.nz)
        smallest_in_row = np.min(prop_vs, axis=1)  # Find the smallest value in each row
        mask = prop_vs[:, 0] == smallest_in_row  # Create a mask for rows where the first element is the smallest
        result = np.all(mask)  # Check if all rows satisfy the condition

        if not result:
            return np.nan, np.nan

         # reshape the model into vs, vp, ra - depth structure
        self.prop_vp = vp.reshape(self.nx*self.ny, self.nz)
        self.prop_vs = vs.reshape(self.nx*self.ny, self.nz)
        self.prop_ra = ra.reshape(self.nx*self.ny, self.nz)
        self.prop_rho = rho.reshape(self.nx*self.ny, self.nz)
        self.prop_von = von.reshape(self.nx*self.ny, self.nz,3)

        # Check differences for each dipping model at each grid on the X-Y plane.
        difflocs = self.check_diff(self.prop_vp, self.prop_vs, self.prop_ra, self.prop_rho, self.prop_von )
        
        # Copy the current RF array for later update in the proposed RF,
        # as not every RF needs to be updated. If the proposed dipping model for a station is the same as the current one,
        # it won't be recalculated.
        self.prop_rf  = copy.deepcopy(self.current_rf)

        # If every dipping model at each grid on the X-Y plane is the same as the previous one, return the current RF array.
        if len(difflocs) == 0:
            return self.obsx, self.prop_rf


       
        # If not, calculate the number of different dipping models for stations that need to be calculated.
        ndiffs = sum(1 for loc in self.staingrididx if loc in difflocs)
        
        t0 = timer.time()

        for i, loc in enumerate(self.staingrididx):
        
            if loc not in difflocs:
                
                continue
    
            
            time, qrf = self.run_raysum_single(i, loc,  **params)

                
            if qrf is np.nan or time is np.nan:
                    
                return np.nan, np.nan
            else:
                start_trace_idx = sum(len(self.obsbaz[idx]) for idx in range(i))
                end_trace_idx = start_trace_idx + len(self.obsbaz[i])
                self.prop_rf[start_trace_idx:end_trace_idx] = qrf


        self.cacl = True
        return self.obsx, self.prop_rf
