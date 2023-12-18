`Learn the Basics <Basics.html>`_ ||
`Quick Start <Quick-Start.html>`_ ||
**Dataset & Data Structure** ||
`Learning Part <Learning.html>`_ ||
`Reasoning Part <Reasoning.html>`_ ||
`Evaluation Metrics <Evaluation.html>`_ ||
`Bridge <Bridge.html>`_


Dataset & Data Structure
========================

In this section, we will look at the datasets and data structures in ABL-Package.

.. code:: python

    # Import necessary libraries and modules
    import torch
    from abl.structures import ListData

Dataset
-------

ABL-Package assumes user data to be structured as a tuple, comprising the following three components:

- ``X``: List[List[Any]]
    A list of sublists representing the input data. We refer to each sublist in ``X`` as an sample and each sample may contain several instances.
- ``gt_pseudo_label``: List[List[Any]], optional
    A list of sublists with each sublist representing a ground truth pseudo label sample. Each sample consists of ground truth pseudo labels for each **instance** within a sample of ``X``. 
- ``Y``: List[Any]
    A list representing the ground truth reasoning result for each **sample** in ``X``.

.. warning::
    Each sublist in ``gt_pseudo_label`` should have the same length as the sublist in ``X``. ``gt_pseudo_label`` is only used to evaluate the performance of the learning part but not to train the model. If the pseudo label of the instances in the datasets are unlabeled, ``gt_pseudo_label`` can be ``None``.

As an illustration, in the MNIST Addition example, the data used for training are organized as follows:

.. image:: ../img/Datasets_1.png
   :width: 350px
   :align: center

Data Structure
--------------

In Abductive Learning, there are various types of data in the training and testing process, such as raw data, pseudo label, index of the pseudo label, abduced pseudo label, etc. To enhance the stability and versatility, ABL-Package uses `abstract data interfaces <../API/abl.structures.html>`_ to encapsulate various data during the implementation of the model.

One of the most commonly used abstract data interface is ``ListData``. Besides orginizing data into tuple, we can also prepare data to be in the form of this data interface.

.. code-block:: python

    # prepare data
    X = [list(torch.randn(3, 28, 28)), list(torch.randn(3, 28, 28))]
    gt_pseudo_label = [[1, 2, 3], [4, 5, 6]]
    Y = [1, 2]

    # convert data into ListData
    data = ListData(X=X, Y=Y, gt_pseudo_label=gt_pseudo_label)

    # get data
    X = data.X
    Y = data.Y
    gt_pseudo_label = data.gt_pseudo_label

    # set data
    data.X = X
    data.Y = Y
    data.gt_pseudo_label = gt_pseudo_label