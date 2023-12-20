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

ABL-Package assumes user data to be either structured as a tuple or a ``ListData`` which is the underlying data structure utilized in the whole package and will be introduced in the next section. Regardless of the chosen format, the data should encompass the following three essential components:

- ``X``: List[List[Any]]
    
    A list of sublists representing the input data. We refer to each sublist in ``X`` as an example and each example may contain several instances.

- ``gt_pseudo_label``: List[List[Any]], optional
    
    A list of sublists with each sublist representing a ground truth pseudo-label example. Each example consists of ground truth pseudo-labels for each **instance** within a example of ``X``. 
    
    .. note::

        ``gt_pseudo_label`` is only used to evaluate the performance of the learning part but not to train the model. If the pseudo-label of the instances in the datasets are unlabeled, ``gt_pseudo_label`` can be ``None``.

- ``Y``: List[Any]
    
    A list representing the ground truth reasoning result for each **example** in ``X``.


.. warning::

    The length of ``X``, ``gt_pseudo_label`` (if not ``None``) and ``Y`` should be the same. Also, each sublist in ``gt_pseudo_label`` should have the same length as the sublist in ``X``.

As an illustration, in the MNIST Addition example, the data used for training are organized as follows:

.. image:: ../img/Datasets_1.png
   :width: 350px
   :align: center

Data Structure
--------------

Besides the user-provided dataset, various forms of data are utilized and dynamicly generate throughout the training and testing process of Abductive Learning framework. Examples include raw data, predicted pseudo-label, abduced pseudo-label, pseudo-label indices, and so on. To manage this diversity and ensure a stable, versatile interface, ABL-Package employs `abstract data interfaces <../API/abl.structures.html>`_ to encapsulate different forms of data that will be used in the total learning process.

``BaseDataElement`` is the base class for all abstract data interfaces. Inherited from ``BaseDataElement``, ``ListData`` is the most commonly used abstract data interface in ABL-Package. As the fundamental data structure, ``ListData`` implements commonly used data manipulation methods and is responsible for transferring data between various components of ABL, ensuring that stages such as prediction, training, and abductive reasoning can utilize ``ListData`` as a unified input format. 

Before proceeding to other stages, user-provided datasets are firstly converted into ``ListData``. For flexibility, ABL-Package also allows user to directly supply data in ``ListData`` format, which similarly requires the inclusion of three attributes: ``X``, ``gt_pseudo_label``, and ``Y``. The following code shows the basic usage of ``ListData``. More information can be found in the `API documentation <../API/abl.structures.html>`_.

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