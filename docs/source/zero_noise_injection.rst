Analysing a zero-noise injection
================================

Below we analyse a zero-noise injection through the :code:`simple-pe-pipe` executable. Rather than generating the frame files ourselves, :code:`simple_pe_pipe` can produce them if we provide an `injection_params.json` file. The following config file can be used:

.. literalinclude:: ../../examples/zero-noise/config.ini
    :language: ini
    :linenos:

and this can be launched with:

.. code-block:: bash

    simple_pe_pipe config.ini

.. note::

    The auxillary data can be downloaded from `here <https://git.ligo.org/stephen-fairhurst/simple-pe/-/tree/main/examples/zero-noise>`_.

Below is a corner plot showing the obtained posterior distribution when using the :code:`IMRPhenomXPHM` waveform model (`Phys.Rev.D 103 (2021) 10,104056 <https://doi.org/10.1103/PhysRevD.103.104056>`_):

.. image:: ./images/IMRPhenomXPHM_injection_corner.png

and below is a corner plot showing the obtained posterior distribution when using the :code:`IMRPhenomTPHM` waveform model (`Phys.Rev.D 105 (2022) 8,084040 <https://doi.org/10.1103/PhysRevD.105.084040>`_):

.. image:: ./images/IMRPhenomTPHM_injection_corner.png

Both of these analyses were performed as part of :code:`simple-pe`'s continuous integration and are therefore using the latest version of the code.
