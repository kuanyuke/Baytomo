#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 11:17:11 2021

@author: kuanyu
"""
import numpy as np
import copy
import math
import scipy.spatial.distance
class Model(object):
    """Handle interpolating methods for a single model vector."""

    @staticmethod
    def kdtree_to_grid(nucleus=None, vp=None, vs=None, ra=None, kdtreeidx = None):
        """convert Voronoi model to grid model
        Parameters:
            nucleus (numpy array): Nucleus array for the model.
            vp (numpy array): P-wave velocity array for the model.
            vs (numpy array): S-wave velocity array for the model.
            ra (numpy array): Density (rho) array for the model.
            kdtreeidx (numpy array): KD-tree indices for conversion.
        Returns:
            tuple: A tuple containing the coverted grid arrays (gridvon, gridvp, gridvs, gridra).
        """        
        gridvon = nucleus[kdtreeidx]
        gridvp= vp[kdtreeidx]
        gridvs= vs[kdtreeidx]
        gridra= ra[kdtreeidx]

        return gridvon, gridvp, gridvs, gridra

    
    @staticmethod
    def get_vsv(vs, ra, methods=None):
        """
        Calculates the vertical shear wave velocity.

        Parameters:
            vs (float): Shear wave velocity.
            ra (float): Radial anisotropy.
            methods (int): Method selection. Default is 1.

        Returns:
            vsv (float): Vertical shear wave velocity.

        Example:
            model = Model()
            vs = 4.0
            ra = 7.0
            methods = 1

            # Calculating vertical shear wave velocity
            vsv = model.get_vsv(vs, ra, methods)
        """

        if methods == None or methods == 1:
            vsv = vs - 0.5 * ra * 0.01 * vs
        elif methods == 2:
            vsv = math.sqrt(np.square(vs)*(1+ra * 0.01))
        elif methods == 3:
            e = (ra + 100) * 0.01
            vsv = math.sqrt(3*(np.square(vs))/(2 + e))
        return vsv

    @staticmethod
    def get_vsh(vs, ra, methods=None):
        """
        Calculates the horizontal shear wave velocity.

        Parameters:
            vs (float): Shear wave velocity.
            ra (float): Radial anisotropy.
            methods (int): Method selection. Default is 1.

        Returns:
            vsv (float): Vertical shear wave velocity.

        Example:
            model = Model()
            vs = 4.0
            ra = 7.0
            methods = 1

            # Calculating vertical shear wave velocity
            vsv = model.get_vsv(vs, ra, methods)
        """
        if methods == None or methods == 1:
            vsh = vs + 0.5 * ra * 0.01 * vs
        elif methods == 2:
            vsh = math.sqrt(np.square(vs) * (1 - ra * 0.01))
        elif methods == 3:
            e = (ra + 100) * 0.01
            vsh = math.sqrt(3*(np.square(vs))/(1+2/e))
        return vsh

    @staticmethod
    def get_vsv_vsh(vs, ra, vs_type=None, methods=None):
        """
        Calculates either vertical or horizontal shear wave velocities for all layers.

        Parameters:
            vs (numpy.ndarray): Array of shear wave velocities.
            ra (numpy.ndarray): Array of Radial anisotropy.
            vs_type (str or int): Shear wave type. 'vsv' or 2 for Rayleigh wave, 'vsh' or 1 for Love wave.
            methods (int): Method selection. Default is None.

        Returns:
            vsv_or_vsh (numpy.ndarray): Array of either vertical or horizontal shear wave velocities for all layers.

        Example:
            model = Model()
            vs = np.array([4.0, 5.0, 6.0])
            ra = np.array([7.0, 8.0, 9.0])
            vs_type = 'vsv'  # or 2 for Rayleigh wave
            methods = 1

            # Calculating either vertical or horizontal shear wave velocities for all layers
            vsv_or_vsh = model.get_vsv_vsh(vs, ra, vs_type, methods)
        """
 

        layers = len(vs)
        if vs_type == 'vsv' or vs_type == 2:  # Rayleigh wave
            vsv = np.zeros(layers)
            for i in range(layers):
                vsv[i] = Model.get_vsv(vs[i], ra[i], methods)
            return vsv
        elif vs_type == 'vsh' or vs_type == 1:  # Love wave
            vsh = np.zeros(layers)
            for i in range(layers):
                vsh[i] = Model.get_vsh(vs[i], ra[i], methods)
            return vsh


    @staticmethod
    def extract_dipping(vons):
        """
        Extracts dipping and strike information from two neighbouring Voronoi cells.

        Parameters:
            vons (list): List of Voronoi points (x, y, z).

        Returns:
            dip (numpy.ndarray): Array containing dip values.
            strike (numpy.ndarray): Array containing strike values.

        Example:
            model = Model()
            vons = [(x1, y1, z1), (x2, y2, z2), ...]

            # Extracting dipping and strike information
            dip, strike = model.extract_dipping(vons)
        """
        layer = len(vons)
        dip = np.zeros(layer)
        strike = np.zeros(layer)
        vons = vons.astype(float)
        for i, von in enumerate(vons):
            if i == 0:
                continue

            # horizontal distance
            dist = scipy.spatial.distance.cdist([vons[i-1]], [von])
            #  vertical distance
            verDistance = abs(vons[i-1][2] -von[2]) 
            dip_radians = math.acos(verDistance/dist) # the result is in radians
            dip_i = math.degrees(dip_radians) # convert the radians to degrees
            
            
            if (von[1] -vons[i-1][1]) == 0 and (von[0] -vons[i-1][0]) == 0:
                azi = 0
                strike_i= azi
            elif (von[1] -vons[i-1][1]) == 0:
                azi = 0
                strike_i= azi
            elif (von[0] -vons[i-1][0]) == 0:
                azi = 90
                strike_i= azi
            else:
                azi = math.atan2(abs(von[1]- vons[i-1][1]),abs(von[0] -vons[i-1][0]))
                strike_i= math.degrees(azi)
            if strike_i < 0:
                strike_i += 360

            dip[i] = dip_i#dip_angle
            strike[i] = strike_i#strike_angle
        
        return dip, strike

    @staticmethod
    def get_stepmodel_from_grids(dz, vp, vs, ra, rho, zmax):
        """
        Return a step-like model from the input model.
        Calculate the thickness.
        vs and ra profiles are based on grids, which means every dz has the
        corresponding vs and ra values. This eliminates dz segments with the same
        vs and ra and rebuilds the profile.
        """
        # Initialize variables
        h = []
        vs_new, ra_new, vp_new, rho_new = [vs[0]], [ra[0]], [vp[0]], [rho[0]]
        thickness = dz  # Current layer thickness accumulator

        # Iterate over vs and ra profiles
        for i in range(1, len(vs)):
            if vs[i] != vs[i - 1] or ra[i] != ra[i - 1]:
                # New layer encountered; append the thickness and reset accumulator
                h.append(thickness)
                thickness = dz
                vs_new.append(vs[i])
                ra_new.append(ra[i])
                vp_new.append(vp[i])
                rho_new.append(rho[i])
            else:
                # Accumulate thickness for the current layer
                thickness += dz

        # Append the last layer
        h.append(thickness)

        # Ensure model has at least two layers
        if len(h) == 1:
            h = [h[0], h[0]]
            vs_new = vs_new * 2
            ra_new = ra_new * 2
            vp_new = vp_new * 2
            rho_new = rho_new * 2

        # Set the last layer's thickness to 0 for compliance
        h[-1] = 0

        return np.array(h), np.array(vp_new), np.array(vs_new), np.array(ra_new), np.array(rho_new)


    @staticmethod
    def get_stepmodel_from_grids_dip(von, dz, vp, vs, ra, rho, zmax):
        """
        Generates a dip model based on input grids and thickness values.

        Parameters:
            von (list): List of points (x, y, z).
            dz (float): Vertical spacing.
            vp (numpy.ndarray): Array of P-wave velocities.
            vs (numpy.ndarray): Array of S-wave velocities.
            ra (numpy.ndarray): Array of density values.
            rho (numpy.ndarray): Array of density values.
            zmax (float): Maximum depth.

        Returns:
            h (numpy.ndarray): Array containing thickness values.
            vp_new (numpy.ndarray): Array of updated P-wave velocities.
            vs_new (numpy.ndarray): Array of updated S-wave velocities.
            ra_new (numpy.ndarray): Array of updated density values.
            rho_new (numpy.ndarray): Array of updated density values.
            dip (numpy.ndarray): Array containing dip values.
            strike (numpy.ndarray): Array containing strike values.
        """
        # Ensure inputs are numpy arrays
        vs = np.array(vs)
        ra = np.array(ra)
        vp = np.array(vp)
        rho = np.array(rho)
        von = np.array(von)

        # Identify change points where vs or ra values differ from the previous
        change_indices = [0] + [
            i for i in range(1, len(vs))
            if vs[i] != vs[i - 1] or ra[i] != ra[i - 1]
        ]

        # Extract unique sections based on change points
        vs_new = vs[change_indices]
        ra_new = ra[change_indices]
        vp_new = vp[change_indices]
        rho_new = rho[change_indices]
        von_new = von[change_indices]

        # Calculate thickness
        h = np.diff(change_indices + [len(vs)]) * dz

        # Calculate dip and strike
        dip, strike = Model.extract_dipping(von_new)

        # Ensure at least two layers
        if len(h) == 1:
            h = np.concatenate((h, h))
            vs_new = np.concatenate((vs_new, vs_new))
            ra_new = np.concatenate((ra_new, ra_new))
            vp_new = np.concatenate((vp_new, vp_new))
            rho_new = np.concatenate((rho_new, rho_new))
            von_new = np.concatenate((von_new, von_new))
            dip = np.zeros(len(h))
            strike = np.zeros(len(h))

        h[-1] = 0

        return h, vp_new, vs_new, ra_new, rho_new, dip, strike

        
    
    @staticmethod
    def create_grid_model( grids):
        """
        create grid model: grid point in km
        - scaling: only use during McMC process

        Parameters:
            grids (dict): Dictionary containing grid parameters.

        Returns:
            grids (dict): Updated grid parameters.

        Example:
            model = Model()
            grids = {'gridx': (0, 100), 'gridy': (0, 50), 'gridz': (0, 10), 'scaling': 0.001, 'nx': 10, 'ny': 5, 'nz': 2}


            # Creating a grid model
            model.create_grid_model(grids)
        """
        grids.xmin, grids.xmax = grids['gridx']
        grids.ymin, grids.ymax = grids['gridy']
        grids.zmin, grids.zmax = grids['gridz']
        
        grids.scaling = grids['scaling']
        grids.zmin = grids.zmin * grids.scaling
        grids.zmax = grids.zmax * grids.scaling


        grids.nx = grids['nx'] 
        grids.ny = grids['ny']
        grids.nz = grids['nz']
        
        grids['nx'] = grids['nx']
        grids['ny'] = grids['ny']     
        
        #!! / in python 3
        grids.dx = float(grids.xmax-grids.xmin)/grids.nx
        grids.dy = float(grids.ymax-grids.ymin)/grids.ny
        grids.dz = float(grids.zmax-grids.zmin)/grids.nz
        grids.mindist= min([grids.dx, grids.dy,grids.dz])



        ##
        x_ = np.linspace((grids.xmin+0.5*grids.dx), (grids.xmax-0.5*grids.dx), grids.nx)
        y_ = np.linspace((grids.ymin+0.5*grids.dy), (grids.ymax-0.5*grids.dy), grids.ny)
        z_ = np.linspace((grids.zmin+0.5*grids.dz), (grids.zmax-0.5*grids.dz), grids.nz)
        grids.gridsmodel = np.vstack(np.meshgrid(x_,y_,z_, indexing='ij')).reshape(3,-1).T
        grids.xyplane = np.vstack(np.meshgrid(x_,y_, indexing='ij')).reshape(2,-1).T
        
        return grids
    @staticmethod
    def sta_nearst_grid(stasloc, grids):
        """
        This is useful for RF to calculate the nearest grid and extract the velocity structure
        at station

        """
        """
        Finds the nearest grid indices for given station locations.

        Parameters:
            stasloc (list): List of station locations (x, y).
            grids (dict): Dictionary containing grid parameters.

        Returns:
            staingrididx (list): List of indices corresponding to the nearest grids for each station.

        Example:
            model = Model().
            stasloc = [(x1, y1), (x2, y2), ...]
            grids = {'gridx': (0, 100), 'gridy': (0, 50), 'gridz': (0, 10), 'scaling': 0.001, 'nx': 10, 'ny': 5, 'nz': 2}

            # Finding nearest grid indices for station locations
            nearest_grid_indices = model.sta_nearst_grid(stasloc, grids)
        """
        
        staingrididx = []
        for i, src in enumerate(stasloc):
            d = ((grids.xyplane-src)**2).sum(axis=1)  
            ndx = d.argsort()
            staingrididx.append(ndx[0])
        return staingrididx
