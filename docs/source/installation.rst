.. role:: raw-latex(raw)
   :format: latex

.. _sec:installation:

  
Installation
=================
Requirements
----------------

============== ==================================
``numpy``      numerical computations
``matplotlib`` plotting library
``pyPdf``      merging PDFs
``configobj``  configuration file
``Cython``     C-extensions for Python
``scipy``      Scientific computations
``joblib``     Parallel processing	
``pandas``     Data manipulation and analysis
============== ==================================


Installation
----------------
(compatible with and 3)

::

   git clone https://github.com/kuanyuke/Baytomo.git
   cd Baytomo
   sudo python setup.py install
   
Resources
----------

-  Algorithm: based on the work of :doc:`Bodin et al., 2012 <references>` and :doc:`Zhang et al., 2018 <references>`.

-  SWD forward modeling: Travel time used the finite-differences method was developed by :doc:`Podvin et al., 1991 <references>`. 

-  SWD forward modeling: ``SURF96`` from Computer Programs in Seismology (:doc:`Herrmann and Ammon, 2002 <references>`). Python wrapper using `pysurf96 <https://github.com/miili/pysurf96>`_ and `SurfTomo <https://github.com/caiweicaiwei/SurfTomo>`_.

-  RF forward modeling: ``rfmini`` (`Joachim Saul, GFZ <https://www.gfz-potsdam.de/en/staff/joachim-saul/>`_).
-  RF forward modeling: Uses ``raysum`` from ([Raysum](https://home.cc.umanitoba.ca/~frederik/Software/)). Python wrapper using [pyfwrd](https://github.com/NoisyLeon/pyfwrd/)

   
The forward modeling codes for surface wave dispersion and receiver functions are already included in the Baytomo package and will be compiled when installing BayHunter. 
