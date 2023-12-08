`Learn the Basics <Basics.html>`_ ||
**Quick Start** ||
`Dataset & Data Structure <Datasets.html>`_ ||
`Learning Part <Learning.html>`_ ||
`Reasoning Part <Reasoning.html>`_ ||
`Evaluation Metrics <Evaluation.html>`_ ||
`Bridge <Bridge.html>`_ 

Quick Start
===========

We use the MNIST Addition task as a quick start example. In this task, the inputs are pairs of MNIST handwritten images, and the outputs are their sums. Refer to the links in each section to dive deeper.

Working with Data
-----------------

ABL-Package assumes data to be in the form of ``(X, gt_pseudo_label, Y)``  where ``X`` is the input of the machine learning model, 
``gt_pseudo_label`` is the ground truth label of each element in ``X`` and ``Y`` is the ground truth reasoning result of each instance in ``X``. 

In the MNIST Addition task, the data loading looks like

.. code:: python

   from examples.mnist_add.datasets.get_mnist_add import get_mnist_add
   
   # train_data and test_data are all tuples consist of X, gt_pseudo_label and Y.
   # If get_pseudo_label is False, gt_pseudo_label will be None
   train_data = get_mnist_add(train=True, get_pseudo_label=True)
   test_data = get_mnist_add(train=False, get_pseudo_label=True)

ABL-Package assumes ``X`` to be of type ``List[List[Any]]``, ``gt_pseudo_label`` can be ``None`` or of the type ``List[List[Any]]`` and ``Y`` should be of type ``List[Any]``. 

.. code:: python

   def describe_structure(lst):
      if not isinstance(lst, list):
         return type(lst).__name__ 
      return [describe_structure(item) for item in lst]
    
   X, gt_pseudo_label, Y = train_data

   print(f"Length of X List[List[Any]]: {len(X)}")
   print(f"Length of gt_pseudo_label List[List[Any]]: {len(gt_pseudo_label)}")
   print(f"Length of Y List[Any]: {len(Y)}\n")

   structure_X = describe_structure(X[:3])
   print(f"Structure of X: {structure_X}")
   structure_gt_pseudo_label = describe_structure(gt_pseudo_label[:3])
   print(f"Structure of gt_pseudo_label: {structure_gt_pseudo_label}")
   structure_Y = describe_structure(Y[:3])
   print(f"Structure of Y: {structure_Y}\n")

   print(f"Shape of X [C, H, W]: {X[0][0].shape}")

Out:

.. code-block:: none
   :class: code-out

   Length of X List[List[Any]]: 30000
   Length of gt_pseudo_label List[List[Any]]: 30000
   Length of Y List[Any]: 30000

   Structure of X: [['Tensor', 'Tensor'], ['Tensor', 'Tensor'], ['Tensor', 'Tensor']]                   
   Structure of gt_pseudo_label: [['int', 'int'], ['int', 'int'], ['int', 'int']]
   Structure of Y: ['int', 'int', 'int']

   Shape of X [C, H, W]: torch.Size([1, 28, 28])


ABL-Package offers several `dataset classes <../API/abl.dataset.html>`_ for different usage, such as ``ClassificationDataset``, ``RegressionDataset`` and ``PredictionDataset``, while users are only required to organize the dataset into the aforementioned format. 

Read more about `preparing datasets <Datasets.html>`_.

Building the Learning Part
--------------------------

To build the machine learning part, we need to wrap our machine learning model into the ``ABLModel`` class. The machine learning model can either be a scikit-learn model or a PyTorch neural network. We use a simple LeNet5 in the MNIST Addition example.

.. code:: python

   from examples.models.nn import LeNet5

   # The number of pseudo labels is 10
   cls = LeNet5(num_classes=10)

Aside from the network, we need to define a criterion, an optimizer, and a device so as to create a ``BasicNN`` object. This class implements ``fit``, ``predict``, ``predict_proba`` and several other methods to enable the PyTorch-based neural network to work as a scikit-learn model.

.. code:: python

   import torch
   from abl.learning import BasicNN

   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(cls.parameters(), lr=0.001, betas=(0.9, 0.99))
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   base_model = BasicNN(cls, criterion, optimizer, device)

.. code:: python

   pred_idx = base_model.predict(X=[torch.randn(1, 28, 28).to(device) for _ in range(32)])
   print(f"Shape of pred_idx : {pred_idx.shape}")
   pred_prob = base_model.predict_proba(X=[torch.randn(1, 28, 28).to(device) for _ in range(32)])
   print(f"Shape of pred_prob : {pred_prob.shape}")

Out:  

.. code-block:: none
   :class: code-out

   Shape of pred_idx : (32,)
   Shape of pred_prob : (32, 10)

Afterward, we wrap the ``base_model`` into ``ABLModel``.

.. code:: python

    from abl.learning import ABLModel

    model = ABLModel(base_model)

Read more about `building the learning part <Learning.html>`_.

Building the Reasoning Part
---------------------------

To build the reasoning part, we first define a knowledge base by
creating a subclass of ``KBBase``, which specifies how to map a pseudo 
label sample to its reasoning result. In the subclass, we initialize the 
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
The reasoner can be used to minimize inconsistencies between the 
knowledge base and the prediction from the learning part. 

.. code:: python

   from abl.reasoning import Reasoner
   
   reasoner = Reasoner(kb)

Read more about `building the reasoning part <Reasoning.html>`_. 


Building Evaluation Metrics
---------------------------

ABL-Package provides two basic metrics, namely ``SymbolMetric`` and ``SemanticsMetric``, which are used to evaluate the accuracy of the machine learning model's predictions and the accuracy of the ``logic_forward`` results, respectively.

.. code:: python

   from abl.evaluation import SemanticsMetric, SymbolMetric

   metric_list = [SymbolMetric(prefix="mnist_add"), SemanticsMetric(kb=kb, prefix="mnist_add")]

Read more about `building evaluation metrics <Evaluation.html>`_

Bridging Learning and Reasoning
---------------------------------------

Now, we use ``SimpleBridge`` to combine learning and reasoning in a unified model.

.. code:: python

   from abl.bridge import SimpleBridge

   bridge = SimpleBridge(model, reasoner, metric_list)

Finally, we proceed with training and testing.

.. code:: python

   bridge.train(train_data, loops=5, segment_size=10000)
   bridge.test(test_data)

Training log would be similar to this:

.. code-block:: none
   :class: code-out
   
   2023/12/02 21:26:57 - abl - INFO - Abductive Learning on the MNIST Addition example.
   2023/12/02 21:32:20 - abl - INFO - Abductive Learning on the MNIST Addition example.
   2023/12/02 21:32:51 - abl - INFO - loop(train) [1/5] segment(train) [1/3] model loss is 1.85589
   2023/12/02 21:32:56 - abl - INFO - loop(train) [1/5] segment(train) [2/3] model loss is 1.50332
   2023/12/02 21:33:02 - abl - INFO - loop(train) [1/5] segment(train) [3/3] model loss is 1.17501
   2023/12/02 21:33:02 - abl - INFO - Evaluation start: loop(val) [1]
   2023/12/02 21:33:07 - abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.350 mnist_add/semantics_accuracy: 0.254 
   2023/12/02 21:33:07 - abl - INFO - Saving model: loop(save) [1]
   2023/12/02 21:33:07 - abl - INFO - Checkpoints will be saved to results/20231202_21_26_57/weights/model_checkpoint_loop_1.pth
   2023/12/02 21:33:13 - abl - INFO - loop(train) [2/5] segment(train) [1/3] model loss is 0.97188
   2023/12/02 21:33:18 - abl - INFO - loop(train) [2/5] segment(train) [2/3] model loss is 0.85622
   2023/12/02 21:33:24 - abl - INFO - loop(train) [2/5] segment(train) [3/3] model loss is 0.81511
   2023/12/02 21:33:24 - abl - INFO - Evaluation start: loop(val) [2]
   2023/12/02 21:33:29 - abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.546 mnist_add/semantics_accuracy: 0.399 
   2023/12/02 21:33:29 - abl - INFO - Saving model: loop(save) [2]
   ...
   2023/12/02 21:34:17 - abl - INFO - loop(train) [5/5] segment(train) [1/3] model loss is 0.03935
   2023/12/02 21:34:23 - abl - INFO - loop(train) [5/5] segment(train) [2/3] model loss is 0.03716
   2023/12/02 21:34:28 - abl - INFO - loop(train) [5/5] segment(train) [3/3] model loss is 0.03346
   2023/12/02 21:34:28 - abl - INFO - Evaluation start: loop(val) [5]
   2023/12/02 21:34:33 - abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.993 mnist_add/semantics_accuracy: 0.986 
   2023/12/02 21:34:33 - abl - INFO - Saving model: loop(save) [5]
   2023/12/02 21:34:33 - abl - INFO - Checkpoints will be saved to results/20231202_21_26_57/weights/model_checkpoint_loop_5.pth
   2023/12/02 21:34:34 - abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.989 mnist_add/semantics_accuracy: 0.978 


Read more about `bridging machine learning and reasoning <Bridge.html>`_.
