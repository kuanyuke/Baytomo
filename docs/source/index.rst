.. Baytomo documentation master file, created by
   sphinx-quickstart on Tue Nov 19 12:25:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview
========

Abstract
--------
To better understand the Earth's interior, many geoscience applications rely on 3D models of Earth's properties to offer a comprehensive view. Equally crucial is the inclusion of uncertainty models, which play a key role in assessing the reliability of these results, particularly in regions with uneven data distribution. Seismic tomography has become a widely employed technique for creating such three-dimensional representations, delivering critical insights into subsurface structures.

Since seismic tomography is inherently nonlinear, advanced techniques are required to deal with its complexity. Among these, Bayes theorem (:doc:`Bayes, 1763 <references>`) has emerged as a preferred approach due to its significant advantages over conventional optimization methods. Instead of focusing on a single best-fit model, Bayesian inversion generates a probability distribution of models, allowing a more thorough assessment of parameter uncertainties and their trade-offs. Transdimensional inversion, enabled by the reversible jump Markov chain Monte Carlo (rj-McMC) algorithm, extends this approach by allowing the model parameterization to dynamically adapt to the data. This flexibility simplifies the parameterization process while maintaining the ability to resolve complex structures. As a result, Bayesian methods provide a powerful framework for exploring solution spaces, visualizing uncertainties, and uncovering the intricacies of geophysical systems.


Baytomo is an open-source Python tool designed to perform Markov Chain Monte Carlo (MCMC) transdimensional Bayesian inversion of surface wave dispersion and/or receiver functions in 3D. In future versions, support for 1D inversion will also be included.

The package builds upon the original 1D version of the software, Bayhunter (:doc:`Dreiling (2019) <references>`), extending its capabilities to handle more complex, three-dimensional Earth structures. This enhancement broadens its applicability, making it a versatile tool for investigating both lateral and depth-dependent variations in subsurface properties.

The algorithm employs a data-driven strategy to infer velocity-depth structures, including 
:math:`\mathrm{V_P}`/:math:`\mathrm{V_S}` ratio, and noise parameters such as data noise correlation and amplitude. Forward modeling codes are provided within the package but are modular, allowing users to easily replace them with their own codes or add entirely different datasets.

This flexibility makes Baytomo a powerful and adaptable tool for geophysical studies, especially for applications requiring detailed velocity modeling and uncertainty quantification.




Available on GitHub: https://github.com/kuanyuke/Baytomo




Documentation
-------------

The documentation to BayHunter contains three chapters, accessible through the navigation menu, and downloadable as `PDF <https://github.com/jenndrei/kuanyuke/blob/master/documentation/Baytomo_v1.1_documentation.pdf>`_. 


- **Installation**: Setting up and running an inversion, targets and parameters, example inversion using synthetic data (minimalistic working example of the code in the appendix)

- **Workflow**: Python framework behind Baytomo, including algorithm and modules

- **Tutorial**: Setting up and running an inversion, targets and parameters, example inversion using synthetic data (minimalistic working example of the code in the appendix)





-------------------------------------------------

.. toctree::
    :maxdepth: 2
    :hidden:
    
    installation
    workflow
    dataformat
    tutorial
    references

Citation
--------


Application examples
--------------------


Comments and Feedback
---------------------

Baytomo is ready to use. It is quick and efficient and I am happy with
the performance. Still, there are always things that can be improved to
make it even faster and more efficient, and user friendlier.

Although we tested the software with a variety of synthetic and real
data, each data set is still unique and shows own characteristics. If
you observe any unforeseen behavior, please share it with me to wipe out
possible problems we haven’t considered.

I am happy to share my experience with you and also if you share your
thoughts with me. I am looking forward to your feedback.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`











