Event Localization
================================

.. _localize:

Overview
--------
The primary way that a GW transient source is localized is through time of
arrival information at the different detectors (see `Triangulation of
gravitational wave sources with a network of detectors <https://doi.org/10
.1088/1367-2630/11/12/123006>`_ for details).  Inclusion of additional
information, particularly relative amplitudes and phases at the different
locations can serve to improve localization and remove some degeneracies (see
`Localization of transient gravitational wave sources: beyond
    triangulation <https://doi.org/10.1088/1361-6382/aab675>`_).  The
:ref:`localization` module implements these analyses.


Example: Localization over the sky
----------------------------------
In this example, we simulate a set of face-on binary neutron star
mergers at a fixed distance and evaluate the localization accuracy for each
event.  The results are plotted as a set of ellipses in the sky.


The analysis is done by the `simple_pe_localization_ellipses` executable.

To see help for this executable please run:

.. code-block:: console

    $ simple_pe_localization_ellipses --help

.. program-output:: simple_pe_localization_ellipses --help

Below is a  plot showing localization ellispes for the default configuration
(advance LIGO, Virgo detectors at design sensitivity):

.. image:: ./images/design_sky_ellipses_found.png


Example: Network sensitivity over the sky
-----------------------------------------
In this example, we calculate the sensitivity of a gravitational-wave
detector network to transient signals at different locations in the
sky.  At each point, we calculate the overall sensitivity as well as
the relative sensitivity of the network to the second GW polarization.

The analysis is done by the `simple_pe_network_coverage`
executable.

To see help for this executable please run:

.. code-block:: console

    $ simple_pe_network_coverage --help

.. program-output:: simple_pe_network_coverage --help

Below is a plot showing network sensitivity for the advanced LIGO,
Virgo network at design, and also a plot of the relative
sensitivity to the second polarization for the network over the sky.

.. image:: ./images/design_sky_sens.png

.. image:: ./images/design_2nd_pol.png
