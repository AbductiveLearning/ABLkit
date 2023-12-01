.. _

Bridge the machine learning and reasoning parts
===============================================

We next need to bridge the machine learning and reasoning parts. In ABL-Package, the ``BaseBridge`` class gives necessary abstract interface definitions to bridge the two parts and ``SimpleBridge`` provides a basic implementation. 
We build a bridge with previously defined ``model``, ``reasoner``, and ``metric_list`` as follows:

.. code:: python

    bridge = SimpleBridge(model, reasoner, metric_list)

``BaseBridge.train`` and ``BaseBridge.test`` trigger the training and testing processes, respectively.

The two methods take the previous prepared ``train_data`` and ``test_data`` as input.

.. code:: python

    bridge.train(train_data)
    bridge.test(test_data)

Aside from data, ``BaseBridge.train`` can also take some other training configs shown as follows:

.. code:: python

    bridge.train(
        # training data
        train_data,
        # number of Abductive Learning loops
        loops=5,
        # data will be divided into segments and each segment will be used to train the model iteratively
        segment_size=10000,
        # evaluate the model every eval_interval loops
        eval_interval=1,
        # save the model every save_interval loops
        save_interval=1,
        # directory to save the model
        save_dir='./save_dir',
    )

In the MNIST Add example, the code to train and test looks like

.. code:: python

    bridge.train(train_data, loops=5, segment_size=10000, save_interval=1, save_dir=weights_dir)
    bridge.test(test_data)
