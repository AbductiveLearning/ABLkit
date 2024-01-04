`Learn the Basics <Basics.html>`_ ||
`Quick Start <Quick-Start.html>`_ ||
`Dataset & Data Structure <Datasets.html>`_ ||
**Learning Part** ||
`Reasoning Part <Reasoning.html>`_ ||
`Evaluation Metrics <Evaluation.html>`_ ||
`Bridge <Bridge.html>`_


Learning Part
=============

In this section, we will look at how to build the learning part. 

In ABL-Package, building the learning part involves two steps:

1. Build a machine learning base model used to make predictions on instance-level data.
2. Instantiate an ``ABLModel`` with the base model, which enables the learning part to process example-level data.

.. code:: python

    import sklearn
    import torchvision
    from abl.learning import BasicNN, ABLModel

Building a base model
---------------------

ABL package allows the base model to be one of the following forms:  

1. Any machine learning model conforming to the scikit-learn style, i.e., models which has implemented the ``fit`` and ``predict`` methods; 

2. A PyTorch-based neural network, provided it has defined the architecture and implemented the ``forward`` method. 

For a scikit-learn model, we can directly use the model itself as a base model. For example, we can customize our base model by a KNN classfier:

.. code:: python

    base_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)

For a PyTorch-based neural network, we need to encapsulate it within a ``BasicNN`` object to create a base model. For example, we can customize our base model by a pre-trained ResNet-18:

.. code:: python

    # Load a PyTorch-based neural network
    cls = torchvision.models.resnet18(pretrained=True)

    # loss function and optimizer are used for training
    loss_fn = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(cls.parameters())

    base_model = BasicNN(cls, loss_fn, optimizer)

BasicNN
^^^^^^^

``BasicNN`` is a wrapper class for PyTorch-based neural networks, which enables them to work as scikit-learn models. It encapsulates the neural network, loss function, optimizer, and other elements into a single object, which can be used as a base model. 

Besides the necessary methods required to instantiate an ``ABLModel``, i.e., ``fit`` and ``predict``, ``BasicNN`` also implements the following methods:

+-------------------------------+------------------------------------------+
| Method                        | Function                                 |
+===============================+==========================================+
| ``train_epoch(data_loader)``  | Train the neural network for one epoch.  |
+-------------------------------+------------------------------------------+
| ``predict_proba(X)``          | Predict the class probabilities of ``X``.|
+-------------------------------+------------------------------------------+
| ``score(X, y)``               | Calculate the accuracy of the model on   |
|                               | test data.                               |
+-------------------------------+------------------------------------------+
| ``save(epoch_id, save_path)`` | Save the model.                          |
+-------------------------------+------------------------------------------+
| ``load(load_path)``           | Load the model.                          |
+-------------------------------+------------------------------------------+

Instantiating an ABLModel
-------------------------

Typically, base model is trained to make predictions on instance-level data, and can not directly process example-level data, which is not suitable for most neural-symbolic tasks. ABL-Package provides the ``ABLModel`` to solve this problem. This class serves as a unified wrapper for all base models, which enables the learning part to train, test, and predict on example-level data.

Generally, we can simply instantiate an ``ABLModel`` by:

.. code:: python

    # Instantiate an ABLModel
    model = ABLModel(base_model)