import numpy as cnp
from numpy cimport ndarray as array

cdef extern from "Python.h":
    object PyComplex_FromDoubles(double, double)

cdef extern from "numpy/arrayobject.h":
    cdef enum PyArray_TYPES:
        PyArray_CHAR, PyArray_UBYTE, PyArray_SBYTE, PyArray_SHORT,
        PyArray_USHORT, PyArray_INT, PyArray_UINT, PyArray_LONG, PyArray_FLOAT,
        PyArray_DOUBLE, PyArray_CFLOAT, PyArray_CDOUBLE, PyArray_OBJECT,
        PyArray_NTYPES, PyArray_NOTYPE

# force inclusion of some header files
#cdef extern from "seispy_unique.h": pass
cdef extern from "numpy/arrayobject.h": pass



cdef extern from "time_2d.h":
    int time_2d(float *HS, float *T, int NX, int NY, float XS, float YS, float EPS_INIT, int MESSAGES)

def time_2d_wrapper(array hs_arr, array tub_arr,
                     int nx, int ny, float xst, float yst, float eps_init, int messages):
    cdef float *hs_ptr
    cdef float *tub_ptr
    #cdef int result

    # Access the raw data buffer
    hs_ptr = <float *> hs_arr.data
    tub_ptr = <float *> tub_arr.data
    
    # Call the C function
    time_2d(hs_ptr, tub_ptr, nx, ny, xst, yst, eps_init, messages)

    return tub_arr

