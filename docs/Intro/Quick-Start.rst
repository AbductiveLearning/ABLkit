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

ABL-Package assumes data to be in the form of ``(X, gt_pseudo_label, Y)``  where ``X`` is the input of the machine learning model, 
``gt_pseudo_label`` is the ground truth label of each element in ``X`` and ``Y`` is the ground truth reasoning result of each instance in ``X``. Note that ``gt_pseudo_label`` is only used to evaluate the performance of the machine learning model but not to train it. If elements in ``X`` are unlabeled, ``gt_pseudo_label`` can be ``None``.

In the MNIST Addition task, the data loading looks like

.. code:: python

   from examples.mnist_add.datasets.get_mnist_add import get_mnist_add
   
   # train_data and test_data both consists of multiple (X, gt_pseudo_label, Y) tuples.
   # If get_pseudo_label is False, gt_pseudo_label in each tuple will be None.
   train_data = get_mnist_add(train=True, get_pseudo_label=True)
   test_data = get_mnist_add(train=False, get_pseudo_label=True)

Read more about `preparing datasets <Datasets.html>`_.

Building the Learning Part
--------------------------

Learnig part is constructed by first defining a machine learning base model and then wrap it into an instance of ``ABLModel`` class. 
The flexibility of ABL package allows the base model to be any machine learning model conforming to the scikit-learn style, which requires implementing the ``fit`` and ``predict`` methods, or a PyTorch-based neural network, provided it has defined the architecture and implemented the ``forward`` method.
In the MNIST Addition example, we build a simple LeNet5 network as the base model.

.. code:: python

   from examples.models.nn import LeNet5

   # The number of pseudo-labels is 10
   cls = LeNet5(num_classes=10)

To facilitate uniform processing, ABL-Package provides the ``BasicNN`` class to convert PyTorch-based neural networks into a format similar to scikit-learn models. To construct a ``BasicNN`` instance, we need also define a loss function, an optimizer, and a device aside from the previous network.

.. code:: python

   import torch
   from abl.learning import BasicNN

   loss_fn = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(cls.parameters(), lr=0.001, betas=(0.9, 0.99))
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   base_model = BasicNN(cls, loss_fn, optimizer, device)

However, Base model built above are trained to make predictions on instance-level data (e.g., a single image), which is not suitable enough for our task. Therefore, we then wrap the ``base_model`` into an instance of ``ABLModel``. This class serves as a unified wrapper for base models, facilitating the learning part to train, test, and predict on example-level data, (e.g., images that comprise the equation).

.. code:: python

    from abl.learning import ABLModel

    model = ABLModel(base_model)

Read more about `building the learning part <Learning.html>`_.

Building the Reasoning Part
---------------------------

To build the reasoning part, we first define a knowledge base by
creating a subclass of ``KBBase``, which specifies how to map a pseudo 
label example to its reasoning result. In the subclass, we initialize the 
``pseudo_label_list`` parameter and override the ``logic_forward`` 
function specifying how to perform (deductive) reasoning.

.. code:: python

   from abl.reasoning import KBBase

   class AddKB(KBBase):
      def __init__(self, pseudo_label_list=list(range(10))):
         super().__init__(pseudo_label_list)

      def logic_forward(self, nums):
         return sum(nums)

   kb = AddKB(pseudo_label_list=list(range(10)))

Then, we create a reasoner by instantiating the class
``Reasoner`` and passing the knowledge base as an parameter.
Due to the indeterminism of abductive reasoning, there could 
be multiple candidates compatible to the knowledge base. 
When this happens, reasoner can minimize inconsistencies between 
the knowledge base and pseudo-labels predicted by the learning part, 
and then return only one candidate that has the highest consistency.

.. code:: python

   from abl.reasoning import Reasoner
   
   reasoner = Reasoner(kb)

Read more about `building the reasoning part <Reasoning.html>`_. 

Building Evaluation Metrics
---------------------------

ABL-Package provides two basic metrics, namely ``SymbolMetric`` and ``ReasoningMetric``, which are used to evaluate the accuracy of the machine learning model's predictions and the accuracy of the ``logic_forward`` results, respectively.

.. code:: python

   from abl.data.evaluation import ReasoningMetric, SymbolMetric

   metric_list = [SymbolMetric(prefix="mnist_add"), ReasoningMetric(kb=kb, prefix="mnist_add")]

Read more about `building evaluation metrics <Evaluation.html>`_

Bridging Learning and Reasoning
---------------------------------------

Now, we use ``SimpleBridge`` to combine learning and reasoning in a unified model.

.. code:: python

   from abl.bridge import SimpleBridge

   bridge = SimpleBridge(model, reasoner, metric_list)

Finally, we proceed with training and testing.

.. code:: python

   bridge.train(train_data, loops=5, segment_size=1/3)
   bridge.test(test_data)

Read more about `bridging machine learning and reasoning <Bridge.html>`_.
