Analysing a GW event
====================
.. _real_event_example:

For a brief overview of the analysis and results on a simulated signal, see
:ref:`zero_noise_injection`.

Here, we analyze a real GW event using the :code:`simple_pe_pipe` executable.
For concreteness, we use events from ER15: S230518h and S230522a (we include
the second example as it was observed in only on detector).

We make use of :code:`simple_pe_pipe` using command line arguments.  For
this, we require the time of event and also the event ID.  Both of these are
available on the grace-db website `S230518h <https://gracedb.ligo
.org/superevents/S230518h/view/>`_ and `S230522a <https://gracedb.ligo
.org/superevents/S230522a/view/>`_.

Example: S230518h
-----------------

This is a two detector event with the two LIGO Observatories, H1 and L1
operational at the time.

To set up the analysis, run :code:`simple_pe_pipe` with the following arguments

.. code-block:: bash

    simple_pe_pipe --sid S230518h \
                   --trigger_time 1368449966.167 \
                   --outdir ./outdir \
                   --approximant IMRPhenomXPHM \
                   --f_low 20 --f_high 1024 \
                   --minimum_data_length 256 \
                   --channels H1:GDS-CALIB_STRAIN_CLEAN \
                               L1:GDS-CALIB_STRAIN_CLEAN \
                   --psd H1:H1_psd.txt L1:L1_psd.txt

Note: Minimum data length is set to be long enough to handle any
type of signal without truncation of the waveform.  In optimizing the SNR,
the code explores the parameter space and may generate BNS waveforms.  If the
event you are running is definitely a BBH merger, then you may find that
reducing the minimum data length speeds up the analysis.

In order to run the analysis, you require the PSDs for the operating
detectors at the time of the event.  These can be obtained from an online
parameter estimation run on the event.  Alternatively, you could use
representative E15 (or O4) noise curves, but that will likely lead to some
systematic effects in the parameter recovery.  [Work is ongoing
to enable simple-pe to generate its own PSDs from the available data round
the event.]

Note: if you wish to submit this job on the LIGO data grid, you will also
need to specify the arguments

.. code-block:: bash

                   --accounting_group
                   --accounting_group_user

Alternatively, you can run from the command line, using the bash script that
is generated and this does not require the specification of the accounting
group.  To do this, run the command

.. code-block:: bash

    bash ./outdir/submit/bash_simple_pe.sh

This will produce the simple-pe results for the event.


Example: S230522a
-----------------

This is a one detector event with the L1 Observatory reported as being
operational at the time of the event.

To set up the analysis, run :code:`simple_pe_pipe` with the following arguments

.. code-block:: bash

    simple_pe_pipe --sid S230522a  \
                   --trigger_time 1368783503.135 \
                   --outdir ./outdir \
                   --approximant IMRPhenomXPHM \
                   --f_low 20 --f_high 1024 \
                   --minimum_data_length 256 \
                   --channels L1:GDS-CALIB_STRAIN_CLEAN \
                   --psd L1:L1_psd.txt

The sid and trigger time are taken from the gracedb page for `S230522a <https://gracedb.ligo
.org/superevents/S230522a/view/>`_.

As before, if you wish to submit the job to run on the LIGO data grid, then
specify the accounting information.  Otherwise, run on the command line with

.. code-block:: bash

    bash ./outdir/submit/bash_simple_pe.sh

Since this is a single detector event, the source is only localized
probabilistically, based upon the instrumental sensitivity.  Consequently,
localization information is not required.  In addition the SNR in the second
polarization is zero by definition.

