Installation
==================

ABL is distributed on `PyPI <https://pypi.org/>`__ and can be installed with ``pip``:

.. code:: console

    # (TODO)
    $ pip install abl

For testing purposes, you can install it using:

.. code:: console

    $ pip install -i https://test.pypi.org/simple/ --extra-index-url https://mirrors.nju.edu.cn/pypi/web/simple/ abl

Alternatively, to install ABL by source code, 
sequentially run following commands in your terminal/command line.

.. code:: console

    $ git clone https://github.com/AbductiveLearning/ABL-Package.git
    $ cd ABL-Package
    $ pip install -v -e .

(Optional) If the use of a :ref:`Prolog-based knowledge base <prolog>` is necessary, the installation of `Swi-Prolog <https://www.swi-prolog.org/>`_ is also required:

For Linux users:

.. code:: console

    $ sudo apt-get install swi-prolog

For Windows and Mac users, please refer to the `Swi-Prolog Download Page <https://www.swi-prolog.org/Download.html>`_.