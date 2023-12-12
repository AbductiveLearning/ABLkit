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
``gt_pseudo_label`` is the ground truth label of each element in ``X`` and ``Y`` is the ground truth reasoning result of each instance in ``X``. Note that ``gt_pseudo_label`` is only used to evaluate the performance of the machine learning part but not to train the model. If elements in ``X`` are unlabeled, ``gt_pseudo_label`` can be ``None``.

In the MNIST Addition task, the data loading looks like

.. code:: python

   from examples.mnist_add.datasets.get_mnist_add import get_mnist_add
   
   # train_data and test_data are all tuples consist of X, gt_pseudo_label and Y.
   # If get_pseudo_label is False, gt_pseudo_label will be None
   train_data = get_mnist_add(train=True, get_pseudo_label=True)
   test_data = get_mnist_add(train=False, get_pseudo_label=True)

ABL-Package assumes ``X`` to be of type ``List[List[Any]]``, ``gt_pseudo_label`` can be ``None`` or of the type ``List[List[Any]]`` and ``Y`` should be of type ``List[Any]``. The following code shows the structure of the dataset used in MNIST Addition.

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


ABL-Package provides several dataset classes for different purposes, including ``ClassificationDataset``, ``RegressionDataset``, and ``PredictionDataset``. However, it's not necessary to encapsulate data into these specific classes. Instead, we only need to structure our datasets in the aforementioned formats.

Read more about `preparing datasets <Datasets.html>`_.

Building the Learning Part
--------------------------

Learnig part is constructed by first defining a base machine learning model and then wrap it into the ``ABLModel`` class. 
The flexibility of ABL package allows the base model to be any machine learning model conforming to the scikit-learn style, which requires implementing the ``fit`` and ``predict`` methods, or a PyTorch-based neural network, provided it has defined the architecture and implemented the ``forward`` method.
In the MNIST Addition example, we build a simple LeNet5 network as the base model.

.. code:: python

   from examples.models.nn import LeNet5

   # The number of pseudo labels is 10
   cls = LeNet5(num_classes=10)

To facilitate uniform processing, ABL-Package provides the ``BasicNN`` class to convert PyTorch-based neural networks into a format similar to scikit-learn models. To construct a ``BasicNN`` instance, we need also define a loss function, an optimizer, and a device aside from the previous network.

.. code:: python

   import torch
   from abl.learning import BasicNN

   loss_fn = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(cls.parameters(), lr=0.001, betas=(0.9, 0.99))
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   base_model = BasicNN(cls, loss_fn, optimizer, device)

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

Afterward, we wrap the scikit-learn style model, ``base_model``, into an instance of ``ABLModel``. This class serves as a unified wrapper for all base models,  facilitating the learning part to train, test, and predict on instance-level data - such as equations in the MNIST Addition.

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

   bridge.train(train_data, loops=5, segment_size=1/3)
   bridge.test(test_data)

Training log would be similar to this:

.. code-block:: none
   :class: code-out

   abl - INFO - Abductive Learning on the MNIST Add example.
   abl - INFO - loop(train) [1/5] segment(train) [1/3] 
   abl - INFO - model loss: 1.91761
   abl - INFO - loop(train) [1/5] segment(train) [2/3] 
   abl - INFO - model loss: 1.59485
   abl - INFO - loop(train) [1/5] segment(train) [3/3] 
   abl - INFO - model loss: 1.33183
   abl - INFO - Evaluation start: loop(val) [1]
   abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.450 mnist_add/semantics_accuracy: 0.237 
   abl - INFO - Saving model: loop(save) [1]
   abl - INFO - Checkpoints will be saved to results/work_dir/weights/model_checkpoint_loop_1.pth
   abl - INFO - loop(train) [2/5] segment(train) [1/3] 
   abl - INFO - model loss: 1.00664
   abl - INFO - loop(train) [2/5] segment(train) [2/3] 
   abl - INFO - model loss: 0.52233
   abl - INFO - loop(train) [2/5] segment(train) [3/3] 
   abl - INFO - model loss: 0.11282
   abl - INFO - Evaluation start: loop(val) [2]
   abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.976 mnist_add/semantics_accuracy: 0.954 
   abl - INFO - Saving model: loop(save) [2]
   abl - INFO - Checkpoints will be saved to results/work_dir/weights/model_checkpoint_loop_2.pth
   ...
   abl - INFO - loop(train) [5/5] segment(train) [1/3] 
   abl - INFO - model loss: 0.04030
   abl - INFO - loop(train) [5/5] segment(train) [2/3] 
   abl - INFO - model loss: 0.03859
   abl - INFO - loop(train) [5/5] segment(train) [3/3] 
   abl - INFO - model loss: 0.03423
   abl - INFO - Evaluation start: loop(val) [5]
   abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.992 mnist_add/semantics_accuracy: 0.984 
   abl - INFO - Saving model: loop(save) [5]
   abl - INFO - Checkpoints will be saved to results/work_dir/weights/model_checkpoint_loop_5.pth
   abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.987 mnist_add/semantics_accuracy: 0.975 

Read more about `bridging machine learning and reasoning <Bridge.html>`_.
