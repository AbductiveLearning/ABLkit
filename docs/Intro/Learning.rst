`Learn the Basics <Basics.html>`_ ||
`Quick Start <Quick-Start.html>`_ ||
`Dataset & Data Structure <Datasets.html>`_ ||
**Learning Part** ||
`Reasoning Part <Reasoning.html>`_ ||
`Evaluation Metrics <Evaluation.html>`_ ||
`Bridge <Bridge.html>`_


Learning Part
=============

Learnig part is constructed by first defining a base machine learning model and then wrap it into an instance of ``ABLModel`` class. 

The flexibility of ABL package allows the base model to be any machine learning model conforming to the scikit-learn style, which requires implementing the ``fit`` and ``predict`` methods, or a PyTorch-based neural network, provided it has defined the architecture and implemented the ``forward`` method. 

Typically, base models are trained and make predictions on instance-level data, e.g. single images in the MNIST dataset, and therefore can not directly utilize sample-level data to train and predict, which is not suitable for most neural-symbolic tasks. ABL-Package provides the ``ABLModel`` to solve this problem. This class serves as a unified wrapper for all base models, which enables the learning part to train, test, and predict on sample-level data. The following two parts shows how to construct an ``ABLModel`` from a scikit-learn model and a PyTorch-based neural network, respectively.

For a scikit-learn model, we can directly use the model to create an instance of ``ABLModel``. For example, we can customize our machine learning model by

.. code:: python

    import sklearn
    from abl.learning import ABLModel

    base_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
    model = ABLModel(base_model)

For a PyTorch-based neural network, we first need to encapsulate it within a ``BasicNN`` object and then use this object to instantiate an instance of ``ABLModel``.  For example, we can customize our machine learning model by

.. code:: python

    # Load a PyTorch-based neural network
    import torchvision
    cls = torchvision.models.resnet18(pretrained=True)

    # loss_fn and optimizer are used for training
    loss_fn = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(cls.parameters())

    base_model = BasicNN(cls, loss_fn, optimizer)
    model = ABLModel(base_model)

Besides ``fit`` and ``predict``, ``BasicNN`` also implements the following methods:

+---------------------------+----------------------------------------+
| Method                    | Function                               |
+===========================+========================================+
| train_epoch(data_loader)  | Train the neural network for one epoch.|
+---------------------------+----------------------------------------+
| predict_proba(X)          | Predict the class probabilities of X.  |
+---------------------------+----------------------------------------+
| score(X, y)               | Calculate the accuracy of the model on |
|                           | test data.                             |
+---------------------------+----------------------------------------+
| save(epoch_id, save_path) | Save the model.                        |
+---------------------------+----------------------------------------+
| load(load_path)           | Load the model.                        |
+---------------------------+----------------------------------------+

