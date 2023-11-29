Use ABL-Package Step by Step
=============================

Using ABL-Package for your learning tasks contains five steps

-  Build the machine learning part
-  Build the reasoning part
-  Build datasets and evaluation metrics
-  Bridge the machine learning and reasoning parts
-  Use ``Bridge.train`` and ``Bridge.test`` to train and test

Build the machine learning part
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    cls = LeNet5(num_classes=len(kb.pseudo_label_list))
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

Build the reasoning part
~~~~~~~~~~~~~~~~~~~~~~~~

Next, we build the reasoning part. In ABL-Package, the reasoning part is wrapped in the ``ReasonerBase`` class. In order to create an instance of this class, we first need to inherit the ``KBBase`` class to customize our knowledge base. Arguments of the ``__init__`` method of the knowledge base should at least contain ``pseudo_label_list`` which is a list of all pseudo labels. The ``logic_forward`` method of ``KBBase`` is an abstract method and we need to instantiate this method in our sub-class to give the ability of deduction to the knowledge base. In general, we can customize our knowledge base by

.. code:: python

    class MyKB(KBBase):
        def __init__(self, pseudo_label_list):
            super().__init__(pseudo_label_list)
        
        def logic_forward(self, *args, **kwargs):
            # Deduction implementation...
            return deduction_result

Aside from the knowledge base, the instantiation of the ``ReasonerBase`` also needs to set an extra argument called ``dist_func``, which is the consistency measure used to select the best candidate from all candidates. In general, we can instantiate our reasoner by

.. code:: python

    kb = MyKB(pseudo_label_list)
    reasoner = ReasonerBase(kb, dist_func="hamming")

In the MNIST Add example, the reasoner looks like

.. code:: python

    class AddKB(KBBase):
        def __init__(self, pseudo_label_list): 
            super().__init__(pseudo_label_list)

        # Implement the deduction function
        def logic_forward(self, nums):
            return sum(nums)

    kb = AddKB(pseudo_label_list=list(range(10)))    
    reasoner = ReasonerBase(kb, dist_func="confidence")

Build datasets and evaluation metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we need to build datasets and evaluation metrics for training and validation. ABL-Package assumes data to be in the form of ``(X, gt_pseudo_label, Y)`` where ``X`` is the input of the machine learning model, ``Y`` is the ground truth of the reasoning result and ``gt_pseudo_label`` is the ground truth label of each element in ``X``. ``X`` should be of type ``List[List[Any]]``, ``Y`` should be of type ``List[Any]`` and ``gt_pseudo_label`` can be ``None`` or of the type ``List[List[Any]]``. 

In the MNIST Add example, the data loading looks like

.. code:: python

    # train_data and test_data are all tuples consist of X, gt_pseudo_label and Y.
    train_data = get_mnist_add(train=True, get_pseudo_label=True)
    test_data = get_mnist_add(train=False, get_pseudo_label=True)

To validate and test the model, we need to inherit from ``BaseMetric`` to define metrics and implement the ``process`` and ``compute_metrics`` methods where the process method accepts a batch of outputs. After processing this batch of data, we save the information to ``self.results`` property. The input results of ``compute_metrics`` is all the information saved in ``process``. Use these information to calculate and return a dict that holds the results of the evaluation metrics. 

We provide two basic metrics, namely ``SymbolMetric`` and ``SemanticsMetric``, which are used to evaluate the accuracy of the machine learning model's predictions and the accuracy of the ``logic_forward`` results, respectively.

In the case of MNIST Add example, the metric definition looks like

.. code:: python

    metric_list = [SymbolMetric(prefix="mnist_add"), SemanticsMetric(kb=kb, prefix="mnist_add")]

Bridge the machine learning and reasoning parts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We next need to bridge the machine learning and reasoning parts. In ABL-Package, the ``BaseBridge`` class gives necessary abstract interface definitions to bridge the two parts and ``SimpleBridge`` provides a basic implementation. 
We build a bridge with previously defined ``model``, ``reasoner``, and ``metric_list`` as follows:

.. code:: python

    bridge = SimpleBridge(model, reasoner, metric_list)

In the MNIST Add example, the bridge creation looks the same.

Use ``Bridge.train`` and ``Bridge.test`` to train and test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
