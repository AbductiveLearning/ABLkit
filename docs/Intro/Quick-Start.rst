`Learn the Basics <Basics.html>`_ ||
**Quick Start** ||
`Dataset & Data Structure <Datasets.html>`_ ||
`Learning Part <Learning.html>`_ ||
`Reasoning Part <Reasoning.html>`_ ||
`Evaluation Metrics <Evaluation.html>`_ ||
`Bridge <Bridge.html>`_ 

Quick Start
===========

We use the MNIST Addition task as a quick start example. In this task, pairs of MNIST handwritten images and their sums are given, alongwith a domain knowledge base which contains information on how to perform addition operations. Our objective is to input a pair of handwritten images and accurately determine their sum. Refer to the links in each section to dive deeper.

Working with Data
-----------------

ABL Kit requires data in the format of ``(X, gt_pseudo_label, Y)``  where ``X`` is a list of input examples containing instances, 
``gt_pseudo_label`` is the ground-truth label of each example in ``X`` and ``Y`` is the ground-truth reasoning result of each example in ``X``. Note that ``gt_pseudo_label`` is only used to evaluate the machine learning model's performance but not to train it.

In the MNIST Addition task, the data loading looks like

.. code:: python

   # The 'datasets' module below is located in 'examples/mnist_add/'
   from datasets import get_dataset
   
   # train_data and test_data are tuples in the format of (X, gt_pseudo_label, Y)
   train_data = get_dataset(train=True)
   test_data = get_dataset(train=False)

Read more about `preparing datasets <Datasets.html>`_.

Building the Learning Part
--------------------------

Learning part is constructed by first defining a base model for machine learning. ABL Kit offers considerable flexibility, supporting any base model that conforms to the scikit-learn style (which requires the implementation of ``fit`` and ``predict`` methods), or a PyTorch-based neural network (which has defined the architecture and implemented ``forward`` method).
In this example, we build a simple LeNet5 network as the base model.

.. code:: python

   # The 'models' module below is located in 'examples/mnist_add/'
   from models.nn import LeNet5

   cls = LeNet5(num_classes=10)

To facilitate uniform processing, ABL Kit provides the ``BasicNN`` class to convert a PyTorch-based neural network into a format compatible with scikit-learn models. To construct a ``BasicNN`` instance, aside from the network itself, we also need to define a loss function, an optimizer, and the computing device.

.. code:: python

   import torch
   from abl.learning import BasicNN

   loss_fn = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.RMSprop(cls.parameters(), lr=0.001)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   base_model = BasicNN(model=cls, loss_fn=loss_fn, optimizer=optimizer, device=device)

The base model built above is trained to make predictions on instance-level data (e.g., a single image), while ABL deals with example-level data. To bridge this gap, we wrap the ``base_model`` into an instance of ``ABLModel``. This class serves as a unified wrapper for base models, facilitating the learning part to train, test, and predict on example-level data, (e.g., images that comprise an equation).

.. code:: python

   from abl.learning import ABLModel

   model = ABLModel(base_model)

Read more about `building the learning part <Learning.html>`_.

Building the Reasoning Part
---------------------------

To build the reasoning part, we first define a knowledge base by creating a subclass of ``KBBase``. In the subclass, we initialize the ``pseudo_label_list`` parameter and override the ``logic_forward`` method, which specifies how to perform (deductive) reasoning that processes pseudo-labels of an example to the corresponding reasoning result. Specifically, for the MNIST Addition task, this ``logic_forward`` method is tailored to execute the sum operation.

.. code:: python

   from abl.reasoning import KBBase

   class AddKB(KBBase):
      def __init__(self, pseudo_label_list=list(range(10))):
         super().__init__(pseudo_label_list)

      def logic_forward(self, nums):
         return sum(nums)

   kb = AddKB()

Next, we create a reasoner by instantiating the class ``Reasoner``, passing the knowledge base as a parameter.
Due to the indeterminism of abductive reasoning, there could be multiple candidate pseudo-labels compatible with the knowledge base. 
In such scenarios, the reasoner can minimize inconsistency and return the pseudo-label with the highest consistency.

.. code:: python

   from abl.reasoning import Reasoner
   
   reasoner = Reasoner(kb)

Read more about `building the reasoning part <Reasoning.html>`_. 

Building Evaluation Metrics
---------------------------

ABL Kit provides two basic metrics, namely ``SymbolAccuracy`` and ``ReasoningMetric``, which are used to evaluate the accuracy of the machine learning model's predictions and the accuracy of the ``logic_forward`` results, respectively.

.. code:: python

   from abl.data.evaluation import ReasoningMetric, SymbolAccuracy

   metric_list = [SymbolAccuracy(), ReasoningMetric(kb=kb)]

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
