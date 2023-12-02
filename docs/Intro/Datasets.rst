`Learn the Basics <Basics.html>`_ ||
`Quick Start <QuickStart.html>`_ ||
**Dataset & Data Structure** ||
`Machine Learning Part <Learning.html>`_ ||
`Reasoning Part <Reasoning.html>`_ ||
`Evaluation Metrics <Evaluation.html>`_ ||
`Bridge <Bridge.html>`_


Dataset & Data Structure
========================

Dataset
-------

ABL-Package offers several `dataset classes <../API/abl.dataset.html>`_ for different usage, such as ``ClassificationDataset``, ``RegressionDataset`` and ``PredictionDataset``, while users are only required to organize the dataset into a tuple consists of the following three components

- ``X``: List[List[Any]]
    A list of instances representing the input data. We refer to each List in ``X`` as an instance and one instance may contain several elements.
- ``gt_pseudo_label``: List[List[Any]], optional
    A list of objects representing the ground truth label of each element in ``X``. It should have the same shape as ``X``. This component is only used to evaluate the performance of the machine learning part but not to train the model. If elements are unlabeled, this component can be ``None``.
- ``Y``: List[Any]
    A list of objects representing the ground truth label of the reasoning result of each instance in ``X``.

In the MNIST Add example, the data used for training looks like:

.. image:: ../img/Datasets_1.png
   :width: 350px
   :align: center

Data Structure
--------------

In Abductive Learning, there are various types of data in the training and testing process, such as raw data, pseudo label, index of the pseudo label, abduced pseudo label, etc. To make the interface stable and possessing good versatility, ABL-Package uses `abstract data interfaces <../API/abl.structures.html>`_ to encapsulate various data during the implementation of the model.

One of the most commonly used abstract data interface is ``ListData``. Besides orginizing data into tuple, we can also prepare data to be in the form of this data interface.

.. code-block:: python

    import torch
    from abl.structures import ListData

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