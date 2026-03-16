.. role:: raw-latex(raw)
   :format: latex

.. _sec:tutorial:

Tutorial
================

This chapter contains the installation instructions of Baytomo,
followed by an example of how to set up and run an inversion.



              
      
Running an inversion
-----------------------------------

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
        
        
Setting up the targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Baytomo provides six target classes (four SWD and two RF), which use two types of forward modeling plugins (``SURF96``,``rfmini``, ``raysum``). For both targets, the user may update the default forward modeling parameters with *set_modelparams*. Parameters and default values are given in :numref:`Table {number} <tab-targetpars>`.
As we have 0


.. _tab-targetpars:
.. table:: Default forward modeling parameters for SWD and RF.

      +-----+----------------+----------------------------------------------+
      | SWD | mode = 1       | 1=fundamental mode, 2=1st higher mode, etc.  |
      +-----+----------------+----------------------------------------------+
      | RF  | gauss = 1.0    | Gauss factor, low pass filter                |
      +-----+----------------+----------------------------------------------+
      |     | water = 0.001  | water level stabilization                    |
      +-----+----------------+----------------------------------------------+
      |     | p = 6.4        | slowness in deg/s                            |
      +-----+----------------+----------------------------------------------+
      |     | nsv = ``None`` | near surface velocity in km/s for computation|
      |     |                | of incident angle (trace rotation). If       |
      |     |                | ``None``, nsv is taken from velocity-model.  |
      +-----+----------------+----------------------------------------------+

If the user wants to implement own forward modeling code, a new forward modeling class for it is needed. After normally initializing a target
with Baytomo, an instance of the new forward modeling class must be initialized and passed to the *update_plugin* method of the target. If
an additional data set is wished to be included in the inversion, i.e., from a non pre-defined target class, a new target class needs to be
implemented, additionally to the forward modeling class that handles the synthetic data computation. For both, the forward modeling class and the
new target class, a template is stored on the GitHub repository. It is important that the classes implement specifically named methods and
parameters to ensure the correct interface with Baytomo.


Initialize the targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The first step towards an inversion is to define the target(s). Therefore, the user needs to pass the observed data to the designated Baytomo target type. Each of the target classes comes with a forward modeling plugin, which is easily exchangeable. For surface waves, a quick Fortran routine based on SURF96 (Herrmann and Ammon, 2002) is pre-installed. For receiver functions, the pre-installed routine is based on rfmini or rayfum. Which forwar mdoelling code will be used for rf calculation depends on if theres back azimuth colum in the station file. Also other targets can be defined.

For surface wave dispersion data, travel times must be assigned to the appropriate target class (`RayleighDispersionPhase`, `RayleighDispersionGroup`, `LoveDispersionPhase`, or `LoveDispersionGroup`), depending on the wave type and observable. These classes support fundamental mode analysis by default. Period-specific uncertainties can also be provided, which influence the weighting during the inversion.

Receiver function data (time-amplitude pairs) must be assigned to the `PReceiverFunction` or `SReceiverFunction` class. Forward modeling for receiver functions can utilize either **rfmini** or **rayfum** routines. The selection of the forward modeling code depends on the presence of back azimuth (`baz`) data in the station file. Parameters such as Gaussian filter width, slowness, water level, and near-surface velocity may need customization for compatibility with the observational data.  

The first step in the inversion workflow is defining the targets. Observational data must be assigned to the corresponding target types provided by the **Baytomo** library. Each target class integrates a forward modeling plugin, allowing flexibility in the choice of forward modeling routines.  



.. code-block:: python
    :linenos:

    from Baytomo import Targets
    
    # Initialize surface wave dispersion target:
    # Requires periods, travel time data, station info, and pair indices.
    swdtarget = Targets.RayleighDispersionPhase(periods, traveltimedata, swdstations, pairs, n_jobs=8)

    # Initialize receiver function target:
    # Requires time, RF data, and station info.
    ftarget = Targets.PReceiverFunction(time, rfdata, rfstations)
    # Optionally, include back-azimuth (baz) and binning information.
    rftarget = Targets.PReceiverFunction(time, rfdata, rfstations, baz=baz, bins=bins)

    # Update receiver function forward modeling parameters (e.g., Gaussian filter, water level, slowness).
    rftarget.moddata.plugin.set_modelparams(gauss=1.0, water=0.001, p=p)
    
    # To perform inversion tasks, individual targets (e.g., SWD or RF) can be combined into a **JointTarget**. The user can define the target configuration based on their inversion objectives. 
    
    # Receiver function only
    targets = Targets.JointTarget(rftargets=[rftarget])
    
    # Surface wave dispersion only
    targets = Targets.JointTarget(swdtargets=[swdtarget])
    
    # Joint inversion (SWD + RF)
    targets = Targets.JointTarget(swdtargets=[swdtarget], rftargets=[rftarget])



Running an inversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To initialize an inversion, first assign the predefined targets, grid, initial parameters, and prior information to the **SingleChain** class. The inversion process is then started using the *SingleChain.run_chain* method, which requires the chain index (`chainidx`) to be specified. 

When running an inversion, it is often advantageous to execute multiple chains simultaneously. This parallel execution ensures better sampling of the parameter space and faster convergence to the global likelihood maximum.

To run chains in parallel, you assign each chain a unique index (`chainidx`) and start them as independent processes. This can be achieved using Python's multiprocessing library or any parallelization framework of your choice.

.. code-block:: python
    :linenos:

    from Baytomo import SingleChain
     
    # Initialize the SingleChain instance with required inputs
    chain = SingleChain(targets, grids, initparams, modelpriors)

    # Set a random seed for reproducibility
    rstate = np.random.RandomState(None)
    random_seed = rstate.randint(1000)

    # Start the inversion for the specified chain index
    chain.run_chain(chainidx, random_seed)
 

The posterior distribution
-----------------------------------

After a chain has finished its iterations, it automatically saves ten
output files in ``.npy`` format (NumPy binary file), holding
:math:`\mathrm{V_S}`-depth models, noise parameters,
:math:`\mathrm{V_P}`/:math:`\mathrm{V_S}` ratios, likelihoods and
misfits for the burn-in (p1) and the posterior sampling phase (p2),
respectively. Every :math:`i`-th chain model is saved to receive a
p2-model collection of :math:`\sim` *maxmodels*, a constraint given by
the user. The files are saved in *savepath/data* as follows:


.. table::
    :width: 70%

    +----------------------+------------------------+
    | ``c*_p1models.npy``  | ``c*_p2models.npy``    |
    | ``c*_p1noise.npy``   | ``c*_p2noise.npy``     |
    | ``c*_p1vpvs.npy``    | ``c*_p2vpvs.npy``      |
    | ``c*_p1likes.npy``   | ``c*_p2likes.npy``     |
    | ``c*_p1misfits.npy`` | ``c*_p2misfits.npy``   |
    +----------------------+------------------------+
    | \*three-digit chain identifier number         |
    +----------------------+------------------------+

While :math:`\mathrm{V_P}`/:math:`\mathrm{V_S}` and the likelihood are
vectors with the lengths defined by *maxmodels*, the models, noise and
misfit values are represented by matrices, additionally dependent on the
maximum number of model layers and the number of targets, both also
defined by the user. The models are saved as Voronoi nuclei
representation. For noise parameters, the matrix contains :math:`r` and
:math:`\sigma` of each target. For the RMS data misfit, the matrix is
composed of the misfit from each target and the joint misfit.

  

Plotting results
-----------------------------------

he inversion results are saved in the specified **savepath**, which contains the configuration file, model outputs, and posterior distribution files. Visualization of these results can be done using the ``PlotFromStorage`` class from ``Baytomo``.

Ensure the following are correctly configured:

1. **Results Path**: Set the path to the directory containing the inversion results.
2. **Configuration File**: Locate the ``test_config.pkl`` file inside the ``data`` folder of the results directory.


+---------------------+-------------------------------------------------+-----------------------------------------------+
| plot_iiter\*        | \* likes, nlayers, noise, vpvs, misfits         | parameter with iterations                     |
|                     |                                                 |                                               |
+---------------------+-------------------------------------------------+-----------------------------------------------+
| plot_posterior\_\*  | \* likes, nlayers, noise, vpvs, misfits         | parameter posterior distribution or           |
|                     |                                                 | :math:`\mathrm{V_S}`-depth models             |
|                     |                                                 |                                               |
+---------------------+-------------------------------------------------+-----------------------------------------------+
| plot_profile\       |                                                 | Velocity and uncertainty 2D Profiles          |
|                     |                                                 |                                               |
|                     |                                                 |                                               |
+---------------------+-------------------------------------------------+-----------------------------------------------+


Example Code
~~~~~~~~~~~~~~~~~

The following Python code demonstrates how to visualize and save different types of results:

.. code-block:: python
   :linenos:

   import os.path as op
   import pandas as pd
   from Baytomo import PlotFromStorage

   # Define the path to results and configuration file
   path = 'results_joint_baz'  # Specify the results directory
   configfile = op.join(path, 'data', 'test_config.pkl')

   # Initialize the PlotFromStorage object
   obj = PlotFromStorage(configfile)

   # Save the final posterior distribution
   obj.save_final_distribution(maxmodels=2000, dev=0.05)

   # Plot iteration-based results
   fig = obj.plot_iiterncells(nchains=50)
   obj.savefig(fig, 'iteration_ncells.png')

   fig = obj.plot_iiterswdnoise(nchains=50, ind=25)
   obj.savefig(fig, 'iteration_swdnoise25.png')

   fig = obj.plot_iiterrfnoise(nchains=50, ind=10)
   obj.savefig(fig, 'iteration_rfnoise.png')

   fig = obj.plot_iitermisfits(nchains=50)
   obj.savefig(fig, 'iteration_misfits.png')

   fig = obj.plot_iiterlikes(nchains=50)
   obj.savefig(fig, 'iteration_likelihood.png')

   # Plot posterior distributions
   fig = obj.plot_posterior_vpvs()
   obj.savefig(fig, 'posterior_vpvs.png')

   fig = obj.plot_posterior_misfits()
   obj.savefig(fig, 'posterior_misfits.png')

   fig = obj.plot_posterior_ncells()
   obj.savefig(fig, 'posterior_ncells.png')

Posterior Noise Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To visualize the posterior noise for each station:

.. code-block:: python

   file = 'obsdata/stations_baz.csv'
   df = pd.read_csv(file, usecols=[2])  

   # Extract unique station names
   stanames = df['name'].tolist()
   figures = obj.plot_posterior_noise(stanames)       
   for i, figure in enumerate(figures):
       obj.savefig(figure, f'posterior_noise_{i}.png')

Plotting 2D Profiles
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``plot_profile`` method allows you to plot 2D model profiles by specifying start and end coordinates.

.. code-block:: python

   fig = obj.plot_profile(x0=23.8, y0=174.5, x1=243, y1=111)

   # Save the figures
   obj.savefig(fig, 'profile1.png')



  
    


