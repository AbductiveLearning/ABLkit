Quick Start
===========

We use the MNIST Add benchmark as a quick start example. In this task, the inputs are 
pairs of MNIST handwritten images, and the outputs are their sums. 
To complete this task, we first process the images through a machine learning model 
to get their corresponding pseudo labels (the number each image represents). 
Then, the recognized labels undergo logical reasoning which calculates their sum. 

Load Data
---------

ABL-Package assumes data to be in the form of ``(X, gt_pseudo_label, Y)`` 
where ``X`` is the input of the machine learning model, 
``Y`` is the ground truth of the reasoning result and 
``gt_pseudo_label`` is the ground truth label of each element in ``X``. 

.. code:: python

    from examples.mnist_add.datasets.get_mnist_add import get_mnist_add

    train_data = get_mnist_add(train=True, get_pseudo_label=True)
    test_data = get_mnist_add(train=False, get_pseudo_label=True)

In the ``get_mnist_add`` above, the return values are tuples of ``(X, gt_pseudo_label, Y)``.

Read more about `prepare datasets <Datasets.html>`_.

Build Machine Learning Models
-----------------------------

We use a simple LeNet5 model to recognize the pseudo labels (numbers) in the images. 
We first build the model and define its corresponding criterion and optimizer for training.

.. code:: python

    import torch
    import torch.nn as nn
    from examples.models.nn import LeNet5

    cls = LeNet5(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cls.parameters(), lr=0.001, betas=(0.9, 0.99))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Afterward, we wrap it in ``ABLModel``.

.. code:: python

    from abl.learning import ABLModel, BasicNN

    base_model = BasicNN(cls, criterion, optimizer, device)
    model = ABLModel(base_model)

Read more about `build machine learning models <Learning.html>`_.

Reasoning (Map pseudo labels to reasoning results)
--------------------------------------------------

First, we build a knowledge base that defines how to deduce 
logical results (i.e., calculate summation) from the pseudo labels 
obtained by machine learning.

.. code:: python

    from abl.reasoning import KBBase, ReasonerBase

    class AddKB(KBBase):
        def __init__(self, pseudo_label_list=list(range(10))):
            super().__init__(pseudo_label_list)

        def logic_forward(self, nums):
            return sum(nums)


    kb = AddKB(pseudo_label_list=list(range(10)))

Then, we define a reasoner, which defines 
how to minimize the inconsistency between the knowledge base and machine learning.

.. code:: python

    reasoner = ReasonerBase(kb, dist_func="confidence")  

Read more about `build the reasoning part <Reasoning.html>`_.  

Bridge Machine Learning and Reasoning
-------------------------------------

Before bridging, we first define the metrics to measure accuracy during validation and testing.

.. code:: python

    from abl.evaluation import SemanticsMetric, SymbolMetric

    metric_list = [SymbolMetric(prefix="mnist_add"), SemanticsMetric(kb=kb, prefix="mnist_add")]


Now, we may use ``SimpleBridge`` to combine machine learning and reasoning together,
setting the stage for subsequent integrated training, validation, and testing.

.. code:: python

    from abl.bridge import SimpleBridge

Finally, we proceed with testing and training.

.. code:: python

    bridge.train(train_data, loops=5, segment_size=10000)
    bridge.test(test_data)

Read more about `defining evaluation metrics <Evaluation.html>`_ and `bridge machine learning and reasoning <Bridge.html>`_.
