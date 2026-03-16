.. role:: raw-latex(raw)
   :format: latex

.. _sec:data:

Input Files
================
This chapter provides an overview of the input file formats and the configuration settings required for Baytomo. Proper preparation and understanding of these files ensure smooth operation and accurate results.



Data format
-----------------------------------
Stations  
~~~~~~~~~~~~~~~~~~~~~~~

For surface wave analysis and receiver function processing using `rfmini` (without considering back azimuth), we use a simplified data format. This format specifies station indices, station names, and their locations in a Cartesian coordinate system.  

The table format is as follows: 

.. container::  
   :name: tab:stations  

   .. table:: Data Format for Stations  
   
      +---------+--------+---------+--------+
      |  idx    |  name  |  sta_x  | sta_y  | 
      +---------+--------+---------+--------+
      |   ...   |   ...  |   ...   |  ...   |
      +---------+--------+---------+--------+
      |   ...   |   ...  |   ...   |  ...   |
      +---------+--------+---------+--------+

If back azimuth considerations are required, additional fields are included in the station data. These fields capture the number of receiver function streams (`num`), back azimuth (`baz`), and the number of bins for the receiver function streams (`bins`). The format is as follows:  

.. container::  
   :name: tab:station-baz  

   .. table:: Station Data Format (With Back Azimuth)
   
      +---------+--------+--------+---------+--------+-------+--------+
      |  idx    |  num   |  name  |  sta_x  | sta_y  |  baz  |  bins  | 
      +---------+--------+--------+---------+--------+-------+--------+
      |   ...   |   ...  |   ...  |   ...   |  ...   | ...   |  ...   |
      +---------+--------+--------+---------+--------+-------+--------+
      |   ...   |   ...  |   ...  |   ...   |  ...   | ...   |  ...   |
      +---------+--------+--------+---------+--------+-------+--------+


Surface wave dispersion 
~~~~~~~~~~~~~~~~~~~~~~~~~
We analyze surface wave dispersion using travel time data between station pairs. In the `ttm` data format, only travel time measurements (`ttm`) are provided. Additional details, such as station pairs and predicted information (`prd`), are stored in separate files. Importantly, the order of pairs and `prd` entries in the `ttm` table matches the order provided in the corresponding pair and `prd` files.  

If no `ttm` data is available for a specific combination, the placeholder value `-1` is used.  

.. container::
   :name: tab:datasurf

   .. table:: Dataformat for SWD.

      +--------+--------------+---------------+--------------+-----+
      |        |    Period 1  |    Period 2   |   Period 3   | ... |
      +--------+--------------+---------------+--------------+-----+
      | Pair 1 |  **ttm11**   |  **ttm12**    | **ttm13**    | ... |
      +--------+--------------+---------------+--------------+-----+
      | Pair 2 |  **ttm21**   |  **ttm22**    | **ttm23**    | ... |
      +--------+--------------+---------------+--------------+-----+
      | Pair 3 |  **ttm31**   |  **ttm32**    | **ttm33**    | ... |
      +--------+--------------+---------------+--------------+-----+
      | Pair 4 |   **-1**     |    **-1**     | **ttm43**    | ... |
      +--------+--------------+---------------+--------------+-----+
      |    .   |     ...      |      ...      |    ...       | ... |
      +--------+--------------+---------------+--------------+-----+
      |    .   |     ...      |      ...      |    ...       | ... |
      +--------+--------------+---------------+--------------+-----+

In the pair format, each station is assigned an index in the station file. We use these indices in the pair file to indicate which two stations are involved in each pair. Even though sometimes we may not have a true source and receiver (e.g., in ambient noise surface wave tomography where the source and receiver are virtual), we still use this format for convenience. The pair file format is shown below:

.. container::
   :name: tab:datapair

   .. table:: Dataformat for pair.

      +-----------+------------+
      |   srcidx  |  rcvidx    |
      +-----------+------------+
      | sta 1 idx | sta 2 idx  |
      +-----------+------------+
      | sta 1 idx | sta 2 idx  |
      +-----------+------------+
      |    ...    |    ...     |
      +-----------+------------+
      |    ...    |    ...     |
      +-----------+------------+

Here, `srcidx` and `rcvidx` represent the indices of the source and receiver stations, respectively, based on the station file.


Receiver functions
~~~~~~~~~~~~~~~~~~~~~~~

In the receiver function)data format, the data corresponds to the order of stations in the station file. If back azimuth information is involved, some stations may have multiple receiver function traces with different back azimuths, but the order will always follow the station file.

.. container::
   :name: tab:datarf

   .. table:: Dataformat for receiver functon.
   
      +----------------------------+-----------+-------------+------------+-----+
      |                            |   Time 1  |    Time 2   |   Time 3   | ... |
      +----------------------------+-----------+-------------+------------+-----+
      | Station 1 (Back azimuth 1) |    ...    |     ...     |     ...    | ... |
      +----------------------------+-----------+-------------+------------+-----+
      | Station 1 (Back azimuth 2) |    ...    |     ...     |     ...    | ... |      
      +----------------------------+-----------+-------------+------------+-----+
      | Station 2                  |    ...    |     ...     |     ...    | ... |
      +----------------------------+-----------+-------------+------------+-----+
      | Station 3                  |    ...    |     ...     |     ...    | ... |
      +----------------------------+-----------+-------------+------------+-----+
      | Station 4                  |    ...    |     ...     |     ...    | ... |
      +----------------------------+-----------+-------------+------------+-----+
      |    ...                     |    ...    |      ...    |    ...     | ... |
      +----------------------------+-----------+-------------+------------+-----+
      |    ...                     |    ...    |      ...    |    ...     | ... |
      +----------------------------+-----------+-------------+------------+-----+
      

Load data 
~~~~~~~~~~~~~~~~~~~~~~~     
Before performing an inversion, the observational data must be prepared according to the described formats. Once the data files are ready, they can be loaded into the program for further processing.  

.. code-block:: python  
    :linenos:

    import numpy as np
    from Baytomo import utils

    # Load travel time data
    traveltimedata = np.loadtxt(ttmfile, delimiter=",")
    
    # Load surface wave dispersion station data
    swdstations = utils.read_stas_dict(swdstasfile, idx='srcidx')
    
    # Load station pair indices
    pairs = utils.read_csv_matrix(pairsfile)
    
    # Load periods
    periods = np.loadtxt(prdsfile)
    
    # Load receiver function time and amplitude data
    time = np.loadtxt(timefile)
    rfdata = np.loadtxt(rffile, delimiter=",")
    
    # Load receiver function station information
    rfstations = utils.read_stas_dict(rfstasfile, idx='srcidx')  
    
    # If back azimuth (baz) and bin information are included:
    rfstations, baz, bins = utils.read_rfstas_dict(rfstasfile, idx='srcidx')
        
              
Setting up parameters (configuration settings)
--------------------------------------------------------

Each chain will be initialized with the targets and with parameter dictionaries. The model grids and model priors that need to be defined are listed with default values in :numref:`Table {number} <tab-gridprior>`, and are explained below in detail.



.. _tab-gridprior:

.. table:: Default model priors and inversion parameters (SI-units, i.e., km, km/s, %). Model prior tuples define the limits (min, max) of a uniform distribution. ``None`` implies that the constraint is not used. Abbreviations and constraints are explained in the text.
   
  
   
   
      +--------------------------------+-----------------------------------------+
      |        grids info              |                 modelpriors             |  
      +------------+-------------------+-----------------------+-----------------+
      | gridx (km) | = (0, 300)        |  ncelss               | = (0, 300)      |
      +------------+-------------------+-----------------------+-----------------+
      | gridy (km) | = (0, 300)        |  Vs (km/s)            | = (3, 5)        |
      +------------+-------------------+-----------------------+-----------------+
      | gridz (km) | = (0, 300)        |  vpvs                 | = (0, 300)      |
      +------------+-------------------+-----------------------+-----------------+
      | nx         | = 60              |  ramethod             | = 3             |    
      +------------+-------------------+-----------------------+-----------------+  
      | ny         | = 60              |  ra (%)               | = (-5,5)        |
      +------------+-------------------+-----------------------+-----------------+
      | nz         | = 60              | :math:`r_{RF}`        | = (0.35, 0.75)  | 
      +------------+-------------------+-----------------------+-----------------+
      | scaling    | = 5               | :math:`\sigma _{RF }` | = (1e-5, 0.05)  |
      +------------+-------------------+-----------------------+-----------------+
      |                                | :math:`frac_{SWD}`    | = 0.            | 
      +--------------------------------+-----------------------+-----------------+
      |                                | :math:`\sigma _{SWD}` | = (1e-5, 0.1)   |
      +--------------------------------+-----------------------+-----------------+
      
The priors include :math:`\mathrm{V_S}`, radial anisotropy, the number of Voronoi cells, and the average crustal :math:`\mathrm{V_P}`/:math:`\mathrm{V_S}`. The ranges given in :numref:`Table {number} <tab-gridprior>` define the bounds of uniform distributions. :math:`\mathrm{V_P}`/:math:`\mathrm{V_S}` can also be specified as a constant value (e.g., 1.73) during the inversion.

The `ramethod` parameter has three options (1, 2, or 3), each corresponding to a different method for calculating radial anisotropy. These methods are as follows:

- **ramethod = 1**:
  - :math:`\mathrm{V_S}^2 = \frac{1}{2} (\mathrm{V_{sv}}^2 + \mathrm{V_{sh}}^2)`
  - :math:`ra = \frac{\mathrm{V_{sh}}^2 - \mathrm{V_{sv}}^2}{2 \mathrm{V_S}^2}`

- **ramethod = 2**:
  - :math:`\mathrm{V_S}^2 = \frac{2 \mathrm{V_{sv}}^2 + \mathrm{V_{sh}}^2}{3}`
  - :math:`ra = \frac{\mathrm{V_{sh}}^2}{\mathrm{V_S}^2}`

- **ramethod = 3**:
  - :math:`\mathrm{V_S}^2 = \frac{2 \mathrm{V_{sv}}^2 + \mathrm{V_{sh}}^2}{3}`
  - :math:`\xi = \left( \frac{\mathrm{V_{sh}}}{\mathrm{V_{sv}}} \right)^2`
  - :math:`RA = (\xi - 1) \times 100\%`


Each noise scaling parameter (:math:`r`, :math:`\sigma`, :math:`frac`) can either be defined by a range (in which case the parameter is inverted during the process) or as a constant value (in which case the parameter remains unchanged throughout the inversion).

For surface wave data, the uncertainty distribution is modeled as a mixture of a Gaussian distribution (with width :math:`\sigma_{SWD}`, representing measurement uncertainties) and a uniform distribution once the Gaussian probability falls below the uniform threshold. The fraction of outliers, :math:`frac_{SWD}`, is defined as the fraction of observations that lie outside the Gaussian distribution. This mixed distribution allows observations with residuals outside the Gaussian to be classified as outliers.

For receiver functions, the assumed correlation law is Gaussian if the RFs are computed using a Gaussian filter, and exponential if an exponential filter is applied. The inversion for :math:`r_{RF}` is feasible for the exponential law, but not for the Gaussian law due to computational constraints. If :math:`r_{RF}` is given as a single value, the Gaussian correlation law is used. Otherwise, if a range for :math:`r_{RF}` is specified, the exponential correlation law is applied. Note that estimating :math:`r_{RF}` using the exponential law during inversion may yield incorrect results if the input RF was filtered using a Gaussian filter.



The inversion parameters can be grouped into three categories: (1) actual inversion parameters, (2) model constraints, and (3) saving options.

.. container::
   :name: tab:mcmcpara

   .. table:: Default inversion parameters (SI-units, i.e., km, km/s, %).  ``None`` implies that the constraint is not used. Abbreviations and constraints are explained in the text.
   


      +------------------------------+
      |        grids info            |  
      +--------------+---------------+
      | iter_burnin  |  100,000      | 
      +--------------+---------------+
      | iter_main    |   50,000      |
      +--------------+---------------+
      | propdist     | = (0.025,     |
      | :math:`^3`   | 0.025, 1,     |
      |              | 1.5  ,0.02,   |
      |              | 0.02, 0.005,  |
      |              | 0.005,0.005)  |            
      +--------------+---------------+
      | acceptance   | = (0, 45)     |
      +--------------+---------------+
      | thinning     | = 10          |
      +--------------+---------------+
      | rcond        | = 1e-6        |
      +--------------+---------------+
      | station      | = 'test'      |
      +--------------+---------------+
      | savepath     | = 'results/'  |
      +--------------+---------------+
      | maxmodels    | = 50000       |
      +--------------+---------------+

Key constraints include the number of iterations during the burn-in and main phases, the initial proposal distribution widths, and the target acceptance rate. 
A **large number of chains** ensures good sampling coverage of the solution space, as each chain starts with a randomly drawn model constrained only by the priors. The **number of iterations** should also be high to promote (though not guarantee) convergence towards the global likelihood maximum. The total number of iterations is given by:

:math:`iter_{total} = iter_{burnin} + iter_{main}`.

It is recommended to allocate more iterations to the burn-in phase (i.e., :math:`iter_{burnin} > iter_{main}`) to increase the likelihood of convergence before entering the posterior exploration phase.



Initial proposal distributions for model modifications are Gaussian, centered at zero, and specified as standard deviations. These values must be provided as a vector of size nine, with the following order:
1. :math:`\mathrm{V_S}`
2. Radial anisotropy
3. Shallow Voronoi cell movements
4. Deep Voronoi cell movements
5. :math:`\mathrm{V_S}` birth/death
6. Radial anisotropy birth/death
7. Surface wave data noise (:math:`\mathrm{SWD}`)
8. Receiver function noise (:math:`\mathrm{RF}`)
9. :math:`\mathrm{V_P}`/:math:`\mathrm{V_S}`

Separate distributions are used for shallow and deep Voronoi cell movements. Cells deeper than half the model depth use the deep distribution, while shallower cells use the shallow distribution.

To enhance sampling efficiency, we implement **adaptive proposal distributions** as described in the Bayhunter tutorial. Without adaptation, the acceptance rate decreases with progress, reducing sampling efficiency. An acceptance rate of ~40–45% is dynamically enforced for each method by adjusting the proposal distribution widths. A minimum standard deviation of 0.001 is maintained for all proposal distributions.

Certain modifications, such as :math:`\mathrm{V_S}` and radial anisotropy, achieve the target acceptance rates with minimal adjustment. However, birth and death steps face significant acceptance challenges; without a minimum distribution width, their standard deviations can shrink to extreme values (e.g., :math:`10^{-10}` km/s or smaller).

Saving Options
~~~~~~~~~~~~~~~~~~~~~~

Saving parameters include :math:`station`, :math:`savepath`, and :math:`maxmodels`:
- **Station**: An optional user-defined label used for reference in automatically saved configuration files after inversion initialization.
- **Savepath**: Specifies the directory where result files are stored. Subfolders include:
  - data: Configuration file, SingleChain output files, combined posterior distribution files, and outlier information.
  - figures: Directory for generated visualizations.

  
This structured approach ensures efficient parameter sampling and detailed result storage, supporting reproducible and interpretable inversion workflows.


It is recommended to allocate more iterations to the burn-in phase (i.e., :math:`iter_{burnin} > iter_{main}`) to increase the likelihood of convergence before entering the posterior exploration phase.      
      
Parameters to constrain the inversion are the number of chains, the number of iterations for the burn-in and the main phase, the initial proposal distribution widths, and the acceptance rate. 
A large number of chains is preferable and assures good coverage of the solution space sampling, as each chain starts with a randomly drawn model only bound by the priors. The number of iterations should also be set high, as it can benefit, but not guarantee, the convergence of the chain towards the global likelihood maximum. 
The total amount of iterations is :math:`iter_{total} = iter_{burnin} + iter_{main}`. We recommend to increase the ratio towards the iterations in the burn-in phase (i.e., :math:`iter_{burnin}>iter_{main}`), so a chain is more likely to have converged when entering the exploration phase for the posterior distribution.

            
The initial proposal distributions, i.e., Gaussian distributions centered at zero, for model modifications, must be given as standard deviations according to each of the model modification methods. The values must be given as a vector of size nine, the order representing following modifications: (1) :math:`\mathrm{V_S}`, (1) radia anisotropy,  (3) move of voronoi cell (shallow), (4) move of voronoi cell (deep) (5) :math:`\mathrm{V_S}` birth/death, (6) ra birth/death, (7) swdnoise, (8) rfnoise, and (9) :math:`\mathrm{V_P}`/:math:`\mathrm{V_S}`.

We utilize different proposal distributions for shallow and deep Voronoi cells. If a Voronoi cell's depth exceeds half the model depth, the deeper proposal distribution is applied; otherwise, the shallow one is used.

To enhance sampling efficiency, we adopt an adaptive proposal distribution approach as introduced in Bayhunter. Without adaptation, the acceptance rate for proposed models typically declines as the inversion progresses, reducing sampling efficiency.

To maintain an efficient exploration of the parameter space, we dynamically adjust the width of each proposal distribution to ensure an acceptance rate of approximately :math:`\sim`\ 40–45 % for each proposal method. A minimum standard deviation of 0.001 is enforced for all proposal distributions. For further details, please refer to the Bayhunter tutorial (:doc:`Dreiling (2019) <references>`).


The saving parameters include the :math:`station`, :math:`savepath` and :math:`maxmodels`. The :math:`station` name is optional and is only used as a reference for the user, for the automatically saved configuration file after initiation of an inversion. 
:math:`savepath` represents the path where all the result files will be stored. A subfolder *data* will contain the configuration file and all the *SingleChain* output files, the combined posterior distribution files and an outlier information file. :math:`savepath` also serves as figure directory.
:math:`maxmodels` is the number of p2-models that will be stored from each chain.



  
    


