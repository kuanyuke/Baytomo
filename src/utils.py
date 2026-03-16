#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 19:06:21 2021

@author: kuanyu
"""
import pandas as pd
import zmq
import pickle
import numpy as np
import os.path as op
import os
import io
from configobj import ConfigObj

def read_csv_matrix(file):
        df = pd.read_csv(file, sep=',')
        return df.values

def read_stas_dict(file,idx):
    """
    Reads a CSV file and structures specific columns into a dictionary format.

    Parameters:
    - file (str): The path to the CSV file to be read.
    - idx (str): The name of the column to be used as the index for the DataFrame.

    Returns:
    - dict: A dictionary where each key is an index value from the specified column (`idx`), and each value is a list of the corresponding row's values from columns 0 and 2.
    """
    df = pd.read_csv(file, usecols=[0, 2, 3])
    return df.set_index(idx).T.to_dict('list') 


def read_rfstas_dict2(file, idx):
    """
    Processes a CSV file containing receiver function station data and returns dictionaries of source coordinates and back-azimuth values.

    Parameters:
    - file (str): The path to the CSV file to be read.
    - idx (str): The column name used to group the data.

    Returns:
    - src_dict (dict): A dictionary where keys are source indices, and values are lists of source coordinates (x and y).
    - baz_dict (dict): A dictionary where keys are source indices, and values are lists of back-azimuth values.
    - bins_list (list): A list containing one bin value per unique source index.
    """

    df = pd.read_csv(file)
    
    src_dict = {}
    baz_dict = {}
    bins_list = []

    for srcidx, group in df.groupby(idx): 
        src_coords = group[['src_x', 'src_y']].iloc[0].tolist()
        baz_values = group['baz'].tolist()
        bins_value = group['bins'].iloc[0]  # Take only the first value

        src_dict[srcidx] = src_coords
        baz_dict[srcidx] = baz_values
        bins_list.append(bins_value)

    return src_dict, baz_dict, bins_list



def read_rfstas_dict(file, idx):
    """
    Processes a CSV file containing receiver function station data and returns dictionaries of source coordinates and back-azimuth values.

    Parameters:
    - file (str): The path to the CSV file to be read.
    - idx (str): The column name used to group the data.

    Returns:
    - src_dict (dict): A dictionary where keys are source indices, and values are lists of source coordinates (x and y).
    - baz_dict (dict): A dictionary where keys are source indices, and values are lists of back-azimuth values.
    - list: A list of bin values extracted from the DataFrame.
    """

    df = pd.read_csv(file)
    
    src_dict = {}
    baz_dict = {}
    
    for srcidx, group  in df.groupby(idx): 
        src_coords = group[['src_x', 'src_y']].iloc[0].tolist()
        baz_values = group['baz'].tolist()
        bins_value = group['bins']  # Assuming bins is constant for each srcidx
        
        src_dict[srcidx] = src_coords
        baz_dict[srcidx] = baz_values
    
    return src_dict, baz_dict,  df['bins'].tolist() 

def read_rfstas_dict0(file, idx):
    df = pd.read_csv(file)
    
    src_dict = {}
    baz_dict = {}
    bins_dict = {}
    
    for srcidx, group in df.groupby(idx): #df.groupby('srcidx'):
        src_coords = group[['src_x', 'src_y']].iloc[0].tolist()
        baz_values = group['baz'].tolist()
        bins_value = group['bins'].iloc[0]  # Assuming bins is constant for each srcidx
        
        src_dict[srcidx] = src_coords
        baz_dict[srcidx] = baz_values
        bins_dict[srcidx] = bins_value

    return src_dict, baz_dict, bins_dict


def get_path(name):
    """
    Constructs and returns the full file path for a given filename within the 'defaults' directory.

    Parameters:
    - name (str): The name of the file to locate.

    Returns:
    - str: The full path to the file.

    Raises:
    - OSError: If the file does not exist.
    """    
    fn = op.join(op.dirname(__file__), 'defaults', name)
    if not op.exists(fn):
        raise OSError('%s does not exist!' % name)
    return fn

def string_decode(section):
    """
    Converts string representations of data in a section to their actual types.

    Parameters:
    - section (dict): A dictionary containing the section data, where some values are string representations of other types.

    Returns:
    - dict: The updated dictionary with converted values.
    """
    keywords = ['station', 'savepath']

    for key in section:
        if key in keywords:
            continue
        try:
            section[key] = eval(section[key])
        except:
            for i, value in enumerate(section[key]):
                section[key][i] = eval(value)
    return section

def save_config(targets, configfile, grids =dict(), priors=dict(),\
                initparams=dict(), mutilparams=dict()):
    """
    Conveniently saves a configfile that you can easily use to view the data
    and parameters used for inversion. This configfile (.pkl) will also be used
    for PlotFromStorage plotting methods after the inversion. With this you can
    redo the plots with the correct data used for the inversion.
    targets: JointTarget instance from inversion
    configfile: outfile name
    priors, initparams: parameter dictionaries important for plotting,
    contains e.g. prior distributions, noise params, iterations etc.
    """
    data = {}
    refs = []

    for target in targets.targets:
        target.get_covariance = None
        ref = target.ref
        refs.append(ref)

    data['targets'] = targets.targets
    data['targetrefs'] = refs
    data['nswdtargets'] = targets.nswdtargets
    data['nrftargets'] = targets.nrftargets
    data['grids'] = grids
    data['priors'] = priors
    data['initparams'] = initparams
    
    with open(configfile, 'wb') as f:
        pickle.dump(data, f)     
        
def load_params(initfile):
    """
    Loads parameters from a configuration file, decoding any string representations of data types.

    Parameters:
    - initfile (str): The path to the configuration file.

    Returns:
    - list: A list of dictionaries, each containing parameters from a section of the configuration file.
    """

    config = ConfigObj(initfile)
    keywords = ['station', 'savepath']
    params = []
    for configsection in config.sections:
        if configsection == 'datapaths':
            continue
        section = config[configsection]
        section = string_decode(section)
        params.append(section)
    return params

def read_config(configfile):
    """
    Reads a pickled configuration file and returns its contents.

    Parameters:
    - configfile (str): The path to the pickled configuration file.

    Returns:
    - dict: The contents of the pickled file.

    Notes:
    - Handles both Python 2 and Python 3 formats for pickled data.
    """    
    try:  # python2
        with open(configfile, 'rb') as f:
            data = pickle.load(f)
    except:  # python3
        with open(configfile, 'rb') as f:
            data = pickle.load(f)

    return data


def _return_c_p_t( filename):
    """Return chainindex, phase number, type of file from filename.
    Only for single chain results.
    """
    c, pt = op.basename(filename).split('.np')[0].split('_')
    cidx = int(c[1:])
    phase, ftype = pt[:2], pt[2:]

    return cidx, phase, ftype

def load( fname,axis=0):
    '''
    Load the whole file, returning all the arrays that were consecutively
    saved on top of each other
    axis defines how the arrays should be concatenated
    '''
    fh = open(fname,'rb')
    fsz = os.fstat(fh.fileno()).st_size
    out = np.load(fh)
    while fh.tell() < fsz:
        out = np.concatenate((out, np.load(fh)), axis=axis)
    return out
    
            
