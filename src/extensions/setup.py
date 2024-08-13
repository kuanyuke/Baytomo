# setup.py
#from setuptools import setup
#from Cython.Build import cythonize

#setup(
#    ext_modules=cythonize("time_2d_wrapper.pyx"),
#)
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Extension module
extensions = [
    Extension(
        "time_2d_wrapper",
        sources=["time_2d_wrapper.pyx"],
        include_dirs=[np.get_include()],
    )
]

# Setup configuration
setup(
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
)


