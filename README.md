# Baytomo

Baytomo is an open-source Python package for performing trans-dimensional Markov Chain Monte Carlo (MCMC) Bayesian inversions of surface wave propagation and/or receiver functions. It supports both 3-D and 1-D inversions, with plans for 1-D inversions in the near future. Baytomo uses a data-driven strategy to solve for the velocity-depth structure, the number of Voronoi cells, the Vp/Vs ratio, and noise parameters such as data noise correlation and amplitude. The package includes forward modeling codes that can be easily replaced by custom implementations and allows the integration of completely different data sets.


## Quick start

### Requirements
* matplotlib
* numpy
* configobj
* zmq (pyzmq)
* Cython
* scipy
* scipy
* joblib
* pandas

### Installation (compatible with Python 3)*

*Although BayHunter is currently compatible with 3.8,
```sh
git clone https://github.com/*******
cd Baytomo
sudo python setup.py install
```

### Documentation and Tutorial

For detailed documentation and usage instructions, more information will be provided.

You can find an example inversion script in the **tutorial folder**. The file tutorial.py includes detailed comments to guide you through its functionality.



### Resources

* Algorithm: based on the work of [Bodin et al., 2012](https://doi.org/10.1029/2011JB008560).
* SWD forward modeling: Uses SURF96 from Computer Programs in Seismology ([CPS](http://www.eas.slu.edu/eqc/eqccps.html)). Python wrapper using [pysurf96](https://github.com/miili/pysurf96) and [SurfTomo](https://github.com/caiweicaiwei/SurfTomo).
* Travel-time calculations: Uses a finite-difference eikonal solver following [Podvin et al., 1999](https://academic.oup.com/gji/article-lookup/doi/10.1111/j.1365-246X.1991.tb03461.x.). The implementation of the eikonal solver from [Podvin et al., 1999](https://academic.oup.com/gji/article-lookup/doi/10.1111/j.1365-246X.1991.tb03461.x.) used in this study is part of the software available at \url{https://doi.org/10.5281/zenodo.11242981} or at \url{https://github.com/andherit/invdfe}. 
* RF forward modeling: Includes **rfmini** from [Joachim Saul, GFZ](https://www.gfz-potsdam.de/en/staff/joachim-saul/).
* RF forward modeling: Uses **raysum** from ([Raysum](https://home.cc.umanitoba.ca/~frederik/Software/)). Python wrapper using [pyfwrd](https://github.com/NoisyLeon/pyfwrd/)



