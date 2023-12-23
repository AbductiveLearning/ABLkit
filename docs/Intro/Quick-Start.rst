`Learn the Basics <Basics.html>`_ ||
**Quick Start** ||
`Dataset & Data Structure <Datasets.html>`_ ||
`Learning Part <Learning.html>`_ ||
`Reasoning Part <Reasoning.html>`_ ||
`Evaluation Metrics <Evaluation.html>`_ ||
`Bridge <Bridge.html>`_ 

Quick Start
===========

We use the MNIST Addition task as a quick start example. In this task, pairs of MNIST handwritten images and their sums are given, alongwith a domain knowledge base which contain information on how to perform addition operations. Our objective is to input a pair of handwritten images and accurately determine their sum. Refer to the links in each section to dive deeper.

Working with Data
-----------------

ABL-Package requires data in the format of ``(X, gt_pseudo_label, Y)``  where ``X`` is a list of input examples containing instances, 
``gt_pseudo_label`` is the ground-truth label of each example in ``X`` and ``Y`` is the ground-truth reasoning result of each example in ``X``. Note that ``gt_pseudo_label`` is only used to evaluate the machine learning model's performance but not to train it. If examples in ``X`` are unlabeled, ``gt_pseudo_label`` should be ``None``.

In the MNIST Addition task, the data loading looks like

.. code:: python

   from examples.mnist_add.datasets.get_mnist_add import get_mnist_add
   
   # train_data and test_data are tuples in the format (X, gt_pseudo_label, Y)
   # If get_pseudo_label is set to False, the gt_pseudo_label in each tuple will be None.
   train_data = get_mnist_add(train=True, get_pseudo_label=True)
   test_data = get_mnist_add(train=False, get_pseudo_label=True)

Read more about `preparing datasets <Datasets.html>`_.

Building the Learning Part
--------------------------

Learnig part is constructed by first defining a base model for machine learning. The ABL-Package offers considerable flexibility, supporting any base model that conforms to the scikit-learn style (which requires the implementation of ``fit`` and ``predict`` methods), or a PyTorch-based neural network (which has defined the architecture and implemented ``forward`` method).
In this example, we build a simple LeNet5 network as the base model.

.. code:: python

   from examples.models.nn import LeNet5

   # The number of pseudo-labels is 10
   cls = LeNet5(num_classes=10)

To facilitate uniform processing, ABL-Package provides the ``BasicNN`` class to convert a PyTorch-based neural network into a format compatible with scikit-learn models. To construct a ``BasicNN`` instance, aside from the network itself, we also need to define a loss function, an optimizer, and the computing device.

.. code:: python

   import torch
   from abl.learning import BasicNN

   loss_fn = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.RMSprop(cls.parameters(), lr=0.001, alpha=0.9)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   base_model = BasicNN(model=cls, loss_fn=loss_fn, optimizer=optimizer, device=device)

The base model built above are trained to make predictions on instance-level data (e.g., a single image), while ABL deals with example-level data. To bridge this gap, we wrap the ``base_model`` into an instance of ``ABLModel``. This class serves as a unified wrapper for base models, facilitating the learning part to train, test, and predict on example-level data, (e.g., images that comprise an equation).

.. code:: python

   from abl.learning import ABLModel

   model = ABLModel(base_model)

Read more about `building the learning part <Learning.html>`_.

Building the Reasoning Part
---------------------------

To build the reasoning part, we first define a knowledge base by creating a subclass of ``KBBase``. In the subclass, we initialize the ``pseudo_label_list`` parameter and override the ``logic_forward`` method, which specifies how to perform (deductive) reasoning that processes pseudo-labels of an example to the corresponding reasoning result.

.. code:: python

   from abl.reasoning import KBBase

   class AddKB(KBBase):
      def __init__(self, pseudo_label_list=list(range(10))):
         super().__init__(pseudo_label_list)

      def logic_forward(self, nums):
         return sum(nums)

   kb = AddKB()

Next, we create a reasoner by instantiating the class ``Reasoner``, passing the knowledge base as a parameter.
Due to the indeterminism of abductive reasoning, there could be multiple candidate pseudo-labels compatible to the knowledge base. 
In such scenarios, the reasoner can minimize inconsistency and return the pseudo-label with the highest consistency.

.. code:: python

   from abl.reasoning import Reasoner
   
   reasoner = Reasoner(kb)

Read more about `building the reasoning part <Reasoning.html>`_. 

Building Evaluation Metrics
---------------------------

ABL-Package provides two basic metrics, namely ``SymbolAccuracy`` and ``ReasoningMetric``, which are used to evaluate the accuracy of the machine learning model's predictions and the accuracy of the ``logic_forward`` results, respectively.

.. code:: python

   from abl.data.evaluation import ReasoningMetric, SymbolAccuracy

   metric_list = [SymbolAccuracy(prefix="mnist_add"), ReasoningMetric(kb=kb, prefix="mnist_add")]

Read more about `building evaluation metrics <Evaluation.html>`_

Bridging Learning and Reasoning
---------------------------------------

Now, we use ``SimpleBridge`` to combine learning and reasoning in a unified ABL framework.

.. code:: python

   from abl.bridge import SimpleBridge

   bridge = SimpleBridge(model, reasoner, metric_list)

Finally, we proceed with training and testing.

.. code:: python

   bridge.train(train_data, loops=1, segment_size=0.01)
   bridge.test(test_data)

Read more about `bridging machine learning and reasoning <Bridge.html>`_.
