=====================================
Simple-PE
=====================================

**simple-pe** is a Python library that performs simple parameter estimation
for gravitational wave signals emitted by coalescing binaries.

The code is a collection of tools which have been developed over the years to
quickly perform parameter estimation on gravitational wave signals, often
using leading order approximations, to obtain accurate, physically
interpretable results.


.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Overview:

    install
    papers
    citing
    genindex
    modindex


.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Examples:

    zero_noise_injection
    GW150914
    localization


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contents:

   cosmology
   detectors
   fstat
   likelihood
   localize
   param_est
   waveforms


.. card:: Perform full parameter estimation on simulated signals
    :link:  zero_noise_injection
    :link-type: ref

    An analysis of simulated signal to produce parameter estimation results
    using the methods described in `arXiv:2304.03731
    <https://arxiv.org/abs/2304.03731>`_.


.. card:: Perform parameter estimation on GW events
    :link:  gw150914_example
    :link-type: ref

    An example analysis of GW159014.


.. card:: Perform source localization
    :link:  localization
    :link-type: ref

    Perform localization using through triangulation, based on the methods
    described in `Triangulation of gravitational wave sources with a network
    of detectors <https://doi.org/10.1088/1367-2630/11/12/123006>`_.


.. card:: Generate Higher order Multipole Amplitudes and SNRs.
    :link:  hm_amps
    :link-type: ref


.. card:: Calculate detector horizon and range plots, including higher modes
    :link:  horizon
    :link-type: ref


.. note::

   This project is under active development



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
