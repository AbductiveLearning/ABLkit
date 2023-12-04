`Learn the Basics <Basics.html>`_ ||
`Quick Start <Quick-Start.html>`_ ||
`Dataset & Data Structure <Datasets.html>`_ ||
**Learning Part** ||
`Reasoning Part <Reasoning.html>`_ ||
`Evaluation Metrics <Evaluation.html>`_ ||
`Bridge <Bridge.html>`_


Learning Part
=============

``ABLModel`` class serves as a unified interface to all machine learning models. Its constructor, the ``__init__`` method, takes a singular argument, ``base_model``. This argument denotes the fundamental machine learning model, which must implement the ``fit`` and ``predict`` methods.

.. code:: python

    class ABLModel:
        def __init__(self, base_model: Any) -> None:
            if not (hasattr(base_model, "fit") and hasattr(base_model, "predict")):
                raise NotImplementedError("The base_model should implement fit and predict methods.")

            self.base_model = base_model

All scikit-learn models satisify this requiremnts, so we can directly use the model to create an instance of ``ABLModel``. For example, we can customize our machine learning model by

.. code:: python

    import sklearn
    from abl.learning import ABLModel

    base_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
    model = ABLModel(base_model)

For a PyTorch-based neural network, we first need to encapsulate it within a ``BasicNN`` object and then use this object to instantiate an instance of ``ABLModel``.  For example, we can customize our machine learning model by

.. code:: python

    # Load a PyTorch-based neural network
    cls = torchvision.models.resnet18(pretrained=True)

    # criterion and optimizer are used for training
    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(cls.parameters())

    base_model = BasicNN(cls, criterion, optimizer)
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

