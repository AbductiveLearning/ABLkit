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

(Optional) If the use of a `Prolog-based knowledge base <prolog>`_ is necessary, you will also need to install Swi-Prolog.

`http://www.swi-prolog.org/build/unix.html <http://www.swi-prolog.org/build/unix.html>`_