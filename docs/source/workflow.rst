.. role:: raw-latex(raw)
   :format: latex

.. _sec:workflow:

Workflow
========

To generate a set of samples from a target density using a transdimensional scheme, we employ a reversible jump Markov Chain Monte Carlo (rj-McMC) algorithm with a dimension-jumping mechanism. This approach extends the Metropolis–Hastings algorithm (:doc:`Mahalanobis (1936) <references>`) by not only perturbing points (Voronoi nuclei) within the current model space but also allowing transitions between state spaces of varying dimensions based on a proposal distribution.

The transdimensional methodology, combined with hierarchical approaches that incorporate error statistics, enables the seismic velocity model's parametrization to be determined directly from the data and prior information. This avoids the need for explicit parametrization choices before the inversion (:doc:`Bodin (2009) <references>`). The algorithm is inspired by the works of :doc:`Bodin (2010) <references>` and :doc:`Zhang (2018) <references>`.

Algorithm Overview
-------------------

Starting from an initial model sampled from the prior distribution of each parameter, the algorithm iteratively performs the following steps:

1. **Propose a new model**:  
   Draw randomly from one of several perturbation types using a normal distribution as the proposal distribution, :math:`q(\mathbf{m}'|\mathbf{m})`. Perturbations can include:
   - Modifying model parameters (:math:`V_s`, etc.)
   - Moving a cell's position
   - Adding or deleting a cell
   - Adjusting data noise or :math:`V_p/V_s`.

2. **Compute the likelihood**:  
   Generate synthetic phase velocities and receiver functions (RFs) for the proposed model and compute its likelihood.

3. **Calculate the acceptance ratio**:  
   Evaluate the acceptance ratio :math:`\alpha` to decide whether to accept the proposed model:

   .. math::
      \alpha(\mathbf{m}'|\mathbf{m}) = \min\left[1, \frac{p(\mathbf{m}')}{p(\mathbf{m})} \cdot 
      \frac{p(\mathbf{d}_{\text{obs}}|\mathbf{m}')}{p(\mathbf{d}_{\text{obs}}|\mathbf{m})} \cdot 
      \frac{q(\mathbf{m}|\mathbf{m}')}{q(\mathbf{m}'|\mathbf{m})}\right]

   If the model is accepted, it replaces the current model; otherwise, the current model remains unchanged.

4. **Repeat**:  
   Continue the process for a defined number of iterations.

Illustration of Workflow
-------------------------

The chain's progression through parameter space is schematically illustrated in :numref:`Figure {number} <fig:bt_flowchart>`. Each chain begins with a current model, and in each iteration, a new model is proposed based on the proposal distribution. The acceptance probability is computed using the prior, proposal, and posterior ratios. 

A proposed model is accepted with its computed acceptance probability. This ensures that even models with lower likelihoods than the current model are occasionally accepted, helping to avoid local maxima. Accepted models replace the current model, while rejected models leave the current model unchanged. This iterative process generates a posterior distribution from all accepted models during the exploration phase.

To facilitate modularity and flexibility, some steps in this workflow are implemented in separate modules, enabling clear separation of responsibilities. For instance, model proposals are handled by sampling.py, likelihood computations by Targets.py, and visualization by Plotting.py. Forward modeling modules are also individual, allowing users to integrate their own forward modeling code. These modular designs ensure ease of maintenance and extendibility.



.. _fig:bt_flowchart:

.. figure :: figures/baytomo_workflow.png

    Schematic workflow of an McMC chain sampling the parameter space. The posterior distribution includes all accepted models of a chain after a chosen number of iterations.
  
    


