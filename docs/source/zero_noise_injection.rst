Analysing a Simulated Signal
============================
.. _zero_noise_injection:

Overview
--------
Simple-pe identifies the observable features in the gravitational-wave signal
and uses them to perform parameter estimation.  In particular, the analysis
identifies the peak of the SNR across the mass and aligned spin space.  It
then uses a quadratic approximation to generate posterior distributions for
those parameters.  Full posteriors for masses, spins, distance and
orientation are generated using this information and also the measured SNR in
additional waveform features: the second GW polarization, higher order
multipoles and precession harmonics.  Details are given in
`Fast inference of binary merger properties using the information
encoded in the gravitational-wave signal <https://doi.org/10.48550/arXiv
.2304.03731>`_

Example: Simulated signal
-------------------------
In this example, we analyze a simulated signal, embedded in zero noise, in
data from the LIGO and Virgo observatories.  The analysis is run using the
:code:`simple-pe-pipe` executable.

The configuration is specified using the configration file

.. literalinclude:: ../../examples/zero-noise/config.ini
    :language: ini
    :linenos:

which can be launched with:

.. code-block:: bash

    simple_pe_pipe config.ini

For this example, the simulated parameters are specified in the file
`injection_params.json` which are used to generate the signal.  The
:code:`trigger_parameters` file contains the starting parameters for the
optimization of the SNR and the :code:`truth` file contains the actual values
--- these are only used to plot the actual parameter values on the final
result plots.  The network is specified by the :code:`asd` dictionary with
instrument sensitivities.

.. note::

    The auxiliary data can be downloaded from `here <https://git.ligo
    .org/stephen-fairhurst/simple-pe/-/tree/main/examples/zero-noise>`_.

Below is a corner plot showing the obtained posterior distribution when
using the :code:`IMRPhenomXPHM` waveform model (`Phys.Rev.D 103 (2021) 10,
104056 <https://doi.org/10.1103/PhysRevD.103.104056>`_):

.. image:: ./images/IMRPhenomXPHM_injection_corner.png

and below is a corner plot showing the obtained posterior distribution when
using the :code:`IMRPhenomTPHM` waveform model (`Phys.Rev.D 105 (2022) 8,
084040 <https://doi.org/10.1103/PhysRevD.105.084040>`_):

.. image:: ./images/IMRPhenomTPHM_injection_corner.png

Both of these analyses were performed as part of :code:`simple-pe`'s
continuous integration and are therefore using the latest version of the code.
