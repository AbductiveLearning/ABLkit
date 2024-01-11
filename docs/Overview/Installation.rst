Installation
==================

Install from PyPI
^^^^^^^^^^^^^^^^^

The easiest way to install ABL kit is using ``pip``:

.. code:: bash

    pip install ablkit

Install from Source
^^^^^^^^^^^^^^^^^^^

Alternatively, to install from source code, 
sequentially run following commands in your terminal/command line.

.. code:: bash

    git clone https://github.com/AbductiveLearning/ABLkit.git
    cd ABLkit
    pip install -v -e .

(Optional) Install SWI-Prolog
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the use of a :ref:`Prolog-based knowledge base <prolog>` is necessary, the installation of `SWI-Prolog <https://www.swi-prolog.org/>`_ is also required:

For Linux users:

.. code:: bash

    sudo apt-get install swi-prolog

For Windows and Mac users, please refer to the `SWI-Prolog Install Guide <https://github.com/yuce/pyswip/blob/master/INSTALL.md>`_.