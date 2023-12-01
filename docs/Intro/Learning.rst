Build the machine learning part
===============================

First, we build the machine learning part, which needs to be wrapped in the ``ABLModel`` class. We can use machine learning models from scikit-learn or based on PyTorch to create an instance of ``ABLModel``. 

- for a scikit-learn model, we can directly use the model to create an instance of ``ABLModel``. For example, we can customize our machine learning model by

  .. code:: python

      # Load a scikit-learn model
      base_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)

      model = ABLModel(base_model)

- for a PyTorch-based neural network, we first need to encapsulate it within a ``BasicNN`` object and then use this object to instantiate an instance of ``ABLModel``.  For example, we can customize our machine learning model by

  .. code:: python

      # Load a PyTorch-based neural network
      cls = torchvision.models.resnet18(pretrained=True)

      # criterion and optimizer are used for training
      criterion = torch.nn.CrossEntropyLoss() 
      optimizer = torch.optim.Adam(cls.parameters())

      base_model = BasicNN(cls, criterion, optimizer)
      model = ABLModel(base_model)

In the MNIST Add example, the machine learning model looks like

.. code:: python

    cls = LeNet5(num_classes=10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cls.parameters(), lr=0.001, betas=(0.9, 0.99))

    base_model = BasicNN(
        cls,
        criterion,
        optimizer,
        device=device,
        batch_size=32,
        num_epochs=1,
    )
    model = ABLModel(base_model)