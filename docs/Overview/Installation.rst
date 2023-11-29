Installation
==================

Case a: If you develop and run ``abl`` directly, install it from source:

.. code-block:: bash

    git clone https://github.com/AbductiveLearning/ABL-Package.git
    cd ABL-Package
    pip install -v -e .
    # "-v" means verbose, or more output
    # "-e" means installing a project in editable mode,
    # thus any local modifications made to the code will take effect without reinstallation.

Case b: If you use ``abl`` as a dependency or third-party package, install it with pip:

.. code-block:: bash

    pip install abl
