============
Installation
============
.. _install:

Install Simple-PE
-----------------

:code:`Simple-PE` is developed and tested for python 3.9+.
We recommend that this code is installed inside a virtual environment using
:code:`virtualenv`. This environment can be installed with python 3.9+ using
`pyenv`_.

.. _pyenv: https://github.com/pyenv/pyenv


Installing Simple-PE from source
--------------------------------

If you would like to install :code:`Simple-PE` from source, then please make
sure that you set up your virtual environment correctly using either the
instructions highlighted above or using your own techniques, you have a
working version of `pip` and you have `git` correctly installed.  We
recommend making use of one of the `IGWN Conda Environments <https://computing
.docs.ligo.org/conda/environments/>`_.

First clone the repository, then install all requirements, then install the
software,

.. code-block:: console

   $ git clone git@git.ligo.org:stephen-fairhurst/simple-pe.git
   $ cd simple-pe/
   $ pip install .