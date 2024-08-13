#!/usr/bin/env python
from numpy.distutils.core import Extension as NumpyExtension
from numpy.distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy



# Fortran extensions
fortran_extensions = [
    NumpyExtension(
        name='Baytomo.surfdisp96_ext',
        sources=['src/extensions/surfdisp96.f'],
        extra_f77_compile_args=['-O3', '-ffixed-line-length-none', '-fbounds-check', '-m64'],
        f2py_options=['only:', 'surfdisp96', ':'],
        language='f77'
    ),
    NumpyExtension(
        name='Baytomo.raysum',
        sources=[
            "src/extensions/raysum/raysum.pyf",
            "src/extensions/raysum/buildmodel.f",
            "src/extensions/raysum/eigenvec.f",
            "src/extensions/raysum/eispack-cg.f",
            "src/extensions/raysum/matrixops.f",
            "src/extensions/raysum/misfit.f",
            "src/extensions/raysum/phaselist.f",
            "src/extensions/raysum/raysum.f",
            "src/extensions/raysum/raysum_interface.f",
            "src/extensions/raysum/readwrite.f",
            "src/extensions/raysum/trace.f",
            "src/extensions/raysum/params.h"
        ],
        extra_f77_compile_args=['-ffixed-line-length-none', '-O3'],
        extra_f90_compile_args=['-O3'],
        language='gfortran'
    ),
]

# Cython extensions
# Define Cython extensions
cython_extensions = cythonize([
    Extension(
        name="Baytomo.rfmini",
        sources=[
            "src/extensions/rfmini/rfmini.pyx",
            "src/extensions/rfmini/greens.cpp",
            "src/extensions/rfmini/model.cpp",
            "src/extensions/rfmini/pd.cpp",
            "src/extensions/rfmini/synrf.cpp",
            "src/extensions/rfmini/wrap.cpp",
            "src/extensions/rfmini/fork.cpp"
        ],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        name="Baytomo.time_2d",
        sources=[
            "src/extensions/time2d/time_2d_wrapper.pyx",
            "src/extensions/time2d/time_2d.c"
        ],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-std=c11']
    )
], language_level="3str")

# Combine all extensions into a single list
extensions = fortran_extensions + cython_extensions

setup (name = 'Baytomo',
       version = '1.0',
       author="Kuan-Yu Ke",
       author_email="kuanyuke@gfz-potsdam.de",
       description = '3-D transdimensional Bayesian inversion for surface wave dispersion and/or receiver functions.',
       packages=['Baytomo'],
       package_dir={
        'Baytomo': 'src'},

       package_data={
        'Baytomo': ['defaults/*'], },
       ext_modules = extensions)

