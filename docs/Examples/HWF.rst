Handwritten Formula (HWF)
=========================

.. raw:: html
    
    <p>For detailed code implementation, please view it on <a class="reference external" href="https://github.com/AbductiveLearning/ABLKit/tree/main/examples/hwf" target="_blank">GitHub</a>.</p>

Below shows an implementation of `Handwritten
Formula <https://arxiv.org/abs/2006.06649>`__. In this
task, handwritten images of decimal formulas and their computed results
are given, alongwith a domain knowledge base containing information on
how to compute the decimal formula. The task is to recognize the symbols
(which can be digits or operators ‘+’, ‘-’, ‘×’, ‘÷’) of handwritten
images and accurately determine their results.

Intuitively, we first use a machine learning model (learning part) to
convert the input images to symbols (we call them pseudo-labels), and
then use the knowledge base (reasoning part) to calculate the results of
these symbols. Since we do not have ground-truth of the symbols, in
Abductive Learning, the reasoning part will leverage domain knowledge
and revise the initial symbols yielded by the learning part through
abductive reasoning. This process enables us to further update the
machine learning model.

.. code:: ipython3

    # Import necessary libraries and modules
    import os.path as osp

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn

    from ablkit.bridge import SimpleBridge
    from ablkit.data.evaluation import ReasoningMetric, SymbolAccuracy
    from ablkit.learning import ABLModel, BasicNN
    from ablkit.reasoning import KBBase, Reasoner
    from ablkit.utils import ABLLogger, print_log

    from datasets import get_dataset
    from models.nn import SymbolNet

Working with Data
-----------------

First, we get the training and testing datasets:

.. code:: ipython3

    train_data = get_dataset(train=True, get_pseudo_label=True)
    test_data = get_dataset(train=False, get_pseudo_label=True)

Both ``train_data`` and ``test_data`` have the same structures: tuples
with three components: X (list where each element is a list of images),
gt_pseudo_label (list where each element is a list of symbols, i.e.,
pseudo-labels) and Y (list where each element is the computed result).
The length and structures of datasets are illustrated as follows.

.. note::

    ``gt_pseudo_label`` is only used to evaluate the performance of
    the learning part but not to train the model.

.. code:: ipython3

    print(f"Both train_data and test_data consist of 3 components: X, gt_pseudo_label, Y")
    print()
    train_X, train_gt_pseudo_label, train_Y = train_data
    print(f"Length of X, gt_pseudo_label, Y in train_data: " +
          f"{len(train_X)}, {len(train_gt_pseudo_label)}, {len(train_Y)}")
    test_X, test_gt_pseudo_label, test_Y = test_data
    print(f"Length of X, gt_pseudo_label, Y in test_data: " +
          f"{len(test_X)}, {len(test_gt_pseudo_label)}, {len(test_Y)}")
    print()
    
    X_0, gt_pseudo_label_0, Y_0 = train_X[0], train_gt_pseudo_label[0], train_Y[0]
    print(f"X is a {type(train_X).__name__}, " +
          f"with each element being a {type(X_0).__name__} of {type(X_0[0]).__name__}.")
    print(f"gt_pseudo_label is a {type(train_gt_pseudo_label).__name__}, " +
          f"with each element being a {type(gt_pseudo_label_0).__name__} " +
          f"of {type(gt_pseudo_label_0[0]).__name__}.")
    print(f"Y is a {type(train_Y).__name__}, " +
          f"with each element being an {type(Y_0).__name__}.")


Out:
    .. code:: none
        :class: code-out

        Both train_data and test_data consist of 3 components: X, gt_pseudo_label, Y
        
        Length of X, gt_pseudo_label, Y in train_data: 10000, 10000, 10000
        Length of X, gt_pseudo_label, Y in test_data: 2000, 2000, 2000
        
        X is a list, with each element being a list of Tensor.
        gt_pseudo_label is a list, with each element being a list of str.
        Y is a list, with each element being an int.
    

The ith element of X, gt_pseudo_label, and Y together constitute the ith
data example. Here we use two of them (the 1001st and the 3001st) as
illstrations:

.. code:: ipython3

    X_1000, gt_pseudo_label_1000, Y_1000 = train_X[1000], train_gt_pseudo_label[1000], train_Y[1000]
    print(f"X in the 1001st data example (a list of images):")
    for i, x in enumerate(X_1000):
        plt.subplot(1, len(X_1000), i+1)
        plt.axis('off') 
        plt.imshow(x.squeeze(), cmap='gray')
    plt.show()
    print(f"gt_pseudo_label in the 1001st data example (a list of ground truth pseudo-labels): {gt_pseudo_label_1000}")
    print(f"Y in the 1001st data example (the computed result): {Y_1000}")
    print()
    X_3000, gt_pseudo_label_3000, Y_3000 = train_X[3000], train_gt_pseudo_label[3000], train_Y[3000]
    print(f"X in the 3001st data example (a list of images):")
    for i, x in enumerate(X_3000):
        plt.subplot(1, len(X_3000), i+1)
        plt.axis('off') 
        plt.imshow(x.squeeze(), cmap='gray')
    plt.show()
    print(f"gt_pseudo_label in the 3001st data example (a list of ground truth pseudo-labels): {gt_pseudo_label_3000}")
    print(f"Y in the 3001st data example (the computed result): {Y_3000}")


Out:
    .. code:: none
        :class: code-out

        X in the 1001st data example (a list of images):
    
    .. image:: ../_static/img/hwf_dataset1.png
        :width: 210px

    .. code:: none
        :class: code-out

        gt_pseudo_label in the 1001st data example (a list of pseudo-labels): ['5', '-', '3']
        Y in the 1001st data example (the computed result): 2
    
    .. code:: none
        :class: code-out

        X in the 3001st data example (a list of images):
    
    .. image:: ../_static/img/hwf_dataset2.png
        :width: 350px

    .. code:: none
        :class: code-out

        gt_pseudo_label in the 3001st data example (a list of pseudo-labels): ['4', '/', '6', '*', '5']
        Y in the 3001st data example (the computed result): 3.333333333333333
    

.. note::

    The symbols in the HWF dataset can be one of digits or operators
    '+', '-', '×', '÷'.

    We may see that, in the 1001st data example, the length of the
    formula is 3, while in the 3001st data example, the length of the
    formula is 5. In the HWF dataset, the lengths of the formulas are 
    1, 3, 5, and 7 (Specifically, 10% of the equations have a length of 1, 
    10% have a length of 3, 20% have a length of 5, and 60% have a length of 7).

Building the Learning Part
--------------------------

To build the learning part, we need to first build a machine learning
base model. We use SymbolNet, and encapsulate it within a ``BasicNN``
object to create the base model. ``BasicNN`` is a class that
encapsulates a PyTorch model, transforming it into a base model with an
sklearn-style interface.

.. code:: ipython3

    # class of symbol may be one of ['1', ..., '9', '+', '-', '*', '/'], total of 14 classes
    cls = SymbolNet(num_classes=13, image_size=(45, 45, 1))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cls.parameters(), lr=0.001, betas=(0.9, 0.99))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    base_model = BasicNN(
        model=cls,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        batch_size=128,
        num_epochs=3,
    )

``BasicNN`` offers methods like ``predict`` and ``predict_prob``, which
are used to predict the class index and the probabilities of each class
for images. As shown below:

.. code:: ipython3

    data_instances = [torch.randn(1, 45, 45) for _ in range(32)]
    pred_idx = base_model.predict(X=data_instances)
    print(f"Predicted class index for a batch of 32 instances: " +
          f"{type(pred_idx).__name__} with shape {pred_idx.shape}")
    pred_prob = base_model.predict_proba(X=data_instances)
    print(f"Predicted class probabilities for a batch of 32 instances: " +
          f"{type(pred_prob).__name__} with shape {pred_prob.shape}")


Out:
    .. code:: none
        :class: code-out

        Predicted class index for a batch of 32 instances: ndarray with shape (32,)
        Predicted class probabilities for a batch of 32 instances: ndarray with shape (32, 14)
    

However, the base model built above deals with instance-level data
(i.e., individual images), and can not directly deal with example-level
data (i.e., a list of images comprising the formula). Therefore, we wrap
the base model into ``ABLModel``, which enables the learning part to
train, test, and predict on example-level data.

.. code:: ipython3

    model = ABLModel(base_model)

As an illustration, consider this example of training on example-level
data using the ``predict`` method in ``ABLModel``. In this process, the
method accepts data examples as input and outputs the class labels and
the probabilities of each class for all instances within these data
examples.

.. code:: ipython3

    from ablkit.data.structures import ListData
    # ListData is a data structure provided by ABL Kit that can be used to organize data examples
    data_examples = ListData()
    # We use the first 1001st and 3001st data examples in the training set as an illustration
    data_examples.X = [X_1000, X_3000]
    data_examples.gt_pseudo_label = [gt_pseudo_label_1000, gt_pseudo_label_3000]
    data_examples.Y = [Y_1000, Y_3000]
    
    # Perform prediction on the two data examples
    # Remind that, in the 1001st data example, the length of the formula is 3, 
    # while in the 3001st data example, the length of the formula is 5.
    pred_label, pred_prob = model.predict(data_examples)['label'], model.predict(data_examples)['prob']
    print(f"Predicted class labels for the 100 data examples: a list of length {len(pred_label)}, \n" +
          f"the first element is a {type(pred_label[0]).__name__} of shape {pred_label[0].shape}, "+
          f"and the second element is a {type(pred_label[1]).__name__} of shape {pred_label[1].shape}.\n")
    print(f"Predicted class probabilities for the 100 data examples: a list of length {len(pred_prob)}, \n"
          f"the first element is a {type(pred_prob[0]).__name__} of shape {pred_prob[0].shape}, " +
          f"and the second element is a {type(pred_prob[1]).__name__} of shape {pred_prob[1].shape}.")


Out:
    .. code:: none
        :class: code-out

        Predicted class labels for the 100 data examples: a list of length 2, 
        the first element is a ndarray of shape (3,), and the second element is a ndarray of shape (5,).
        
        Predicted class probabilities for the 100 data examples: a list of length 2, 
        the first element is a ndarray of shape (3, 14), and the second element is a ndarray of shape (5, 14).
    

Building the Reasoning Part
---------------------------

In the reasoning part, we first build a knowledge base which contains
information on how to compute a formula. We build it by
creating a subclass of ``KBBase``. In the derived subclass, we
initialize the ``pseudo_label_list`` parameter specifying list of
possible pseudo-labels, and override the ``logic_forward`` function
defining how to perform (deductive) reasoning.

.. code:: ipython3

    class HwfKB(KBBase):
        def __init__(self, pseudo_label_list=["1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/"]):
            super().__init__(pseudo_label_list)
    
        def _valid_candidate(self, formula):
            if len(formula) % 2 == 0:
                return False
            for i in range(len(formula)):
                if i % 2 == 0 and formula[i] not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    return False
                if i % 2 != 0 and formula[i] not in ["+", "-", "*", "/"]:
                    return False
            return True
        
        # Implement the deduction function
        def logic_forward(self, formula):
            if not self._valid_candidate(formula):
                return np.inf
            return eval("".join(formula))
    
    kb = HwfKB()

The knowledge base can perform logical reasoning (both deductive
reasoning and abductive reasoning). Below is an example of performing
(deductive) reasoning, and users can refer to :ref:`Performing abductive 
reasoning in the knowledge base <kb-abd>` for details of abductive reasoning.

.. code:: ipython3

    pseudo_labels = ["1", "-", "2", "*", "5"]
    reasoning_result = kb.logic_forward(pseudo_labels)
    print(f"Reasoning result of pseudo-labels {pseudo_labels} is {reasoning_result}.")


Out:
    .. code:: none
        :class: code-out

        Reasoning result of pseudo-labels ['1', '-', '2', '*', '5'] is -9.
    

.. note::

    In addition to building a knowledge base based on ``KBBase``, we
    can also establish a knowledge base with a ground KB using ``GroundKB``.
    The corresponding code can be found in the ``examples/hwf/main.py`` file. Those
    interested are encouraged to examine it for further insights.

    Also, when building the knowledge base, we can also set the
    ``max_err`` parameter during initialization, which is shown in the
    ``examples/hwf/main.py`` file. This parameter specifies the upper tolerance limit
    when comparing the similarity between the reasoning result of pseudo-labels and 
    the ground truth during abductive reasoning, with a default
    value of 1e-10.

Then, we create a reasoner by instantiating the class ``Reasoner``. Due
to the indeterminism of abductive reasoning, there could be multiple
candidates compatible with the knowledge base. When this happens, reasoner
can minimize inconsistencies between the knowledge base and
pseudo-labels predicted by the learning part, and then return only one
candidate that has the highest consistency.

.. code:: ipython3

    reasoner = Reasoner(kb)

.. note::

    During creating reasoner, the definition of “consistency” can be
    customized within the ``dist_func`` parameter. In the code above, we
    employ a consistency measurement based on confidence, which calculates
    the consistency between the data example and candidates based on the
    confidence derived from the predicted probability. In ``examples/hwf/main.py``, we
    provide options for utilizing other forms of consistency measurement.

    Also, during the process of inconsistency minimization, we can
    leverage `ZOOpt library <https://github.com/polixir/ZOOpt>`__ for
    acceleration. Options for this are also available in ``examples/hwf/main.py``. Those
    interested are encouraged to explore these features.

Building Evaluation Metrics
---------------------------

Next, we set up evaluation metrics. These metrics will be used to
evaluate the model performance during training and testing.
Specifically, we use ``SymbolAccuracy`` and ``ReasoningMetric``, which are
used to evaluate the accuracy of the machine learning model’s
predictions and the accuracy of the final reasoning results,
respectively.

.. code:: ipython3

    metric_list = [SymbolAccuracy(prefix="hwf"), ReasoningMetric(kb=kb, prefix="hwf")]

Bridging Learning and Reasoning
-------------------------------

Now, the last step is to bridge the learning and reasoning part. We
proceed with this step by creating an instance of ``SimpleBridge``.

.. code:: ipython3

    bridge = SimpleBridge(model, reasoner, metric_list)

Perform training and testing by invoking the ``train`` and ``test``
methods of ``SimpleBridge``.

.. code:: ipython3

    # Build logger
    print_log("Abductive Learning on the HWF example.", logger="current")
    log_dir = ABLLogger.get_current_instance().log_dir
    weights_dir = osp.join(log_dir, "weights")
    
    bridge.train(train_data, loops=3, segment_size=1000, save_dir=weights_dir)
    bridge.test(test_data)

The log will appear similar to the following:

Log:
    .. code:: none
        :class: code-out

        abl - INFO - Abductive Learning on the HWF example.
        abl - INFO - loop(train) [1/3] segment(train) [1/10] 
        abl - INFO - model loss: 0.00024
        abl - INFO - loop(train) [1/3] segment(train) [2/10] 
        abl - INFO - model loss: 0.00011
        abl - INFO - loop(train) [1/3] segment(train) [3/10] 
        abl - INFO - model loss: 0.00332
        ...
        abl - INFO - Eval start: loop(val) [1]
        abl - INFO - Evaluation ended, hwf/character_accuracy: 0.997 hwf/reasoning_accuracy: 0.985 
        abl - INFO - loop(train) [2/3] segment(train) [1/10] 
        abl - INFO - model loss: 0.00126
        ...
        abl - INFO - Eval start: loop(val) [2]
        abl - INFO - Evaluation ended, hwf/character_accuracy: 0.998 hwf/reasoning_accuracy: 0.989 
        abl - INFO - loop(train) [3/3] segment(train) [1/10] 
        abl - INFO - model loss: 0.00030
        ...
        abl - INFO - Eval start: loop(val) [3]
        abl - INFO - Evaluation ended, hwf/character_accuracy: 0.999 hwf/reasoning_accuracy: 0.996 
        abl - INFO - Test start:
        abl - INFO - Evaluation ended, hwf/character_accuracy: 0.997 hwf/reasoning_accuracy: 0.986

Performance
-----------

We present the results of ABL as follows, which include the reasoning accuracy (for different equation lengths in the HWF dataset), and the training time (to achieve the accuracy using all equation lengths). These results are compared with the following methods:

- `NGS <https://github.com/liqing-ustc/NGS>`_: A neural-symbolic framework that uses a grammar model and a back-search algorithm to improve its computing process;

- `DeepProbLog <https://github.com/ML-KULeuven/deepproblog/tree/master>`_: An extension of ProbLog by introducing neural predicates in Probabilistic Logic Programming;

- `DeepStochLog <https://github.com/ML-KULeuven/deepstochlog/tree/main>`_: A neural-symbolic framework based on stochastic logic program.

.. raw:: html

    <style type="text/css">
    .tg  {border-collapse:collapse;border-spacing:0;margin-bottom:20px;}
    .tg td, .tg th {border:1px solid #ddd;padding:10px 15px;text-align:center;}
    .tg th {background-color:#f5f5f5;color:#333333;}
    .tg tr:nth-child(even) {background-color:#f9f9f9;}
    .tg tr:nth-child(odd) {background-color:#ffffff;}
    </style>
    <table class="tg" style="margin-left: auto; margin-right: auto;">
    <thead>
    <tr>
        <th rowspan="2"></th>
        <th colspan="5">Reasoning Accuracy<br><span style="font-weight: normal; font-size: smaller;">(for different equation lengths)</span></th>
        <th rowspan="2">Training Time (s)<br><span style="font-weight: normal; font-size: smaller;">(to achieve the Acc. using all lengths)</span></th>
    </tr>
    <tr>
        <th>1</th>
        <th>3</th>
        <th>5</th>
        <th>7</th>
        <th>All</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>NGS</td>
        <td>91.2</td>
        <td>89.1</td>
        <td>92.7</td>
        <td>5.2</td>
        <td>98.4</td>
        <td>426.2</td>
    </tr>
    <tr>
        <td>DeepProbLog</td>
        <td>90.8</td>
        <td>85.6</td>
        <td>timeout*</td>
        <td>timeout</td>
        <td>timeout</td>
        <td>timeout</td>
    </tr>
    <tr>
        <td>DeepStochLog</td>
        <td>92.8</td>
        <td>87.5</td>
        <td>92.1</td>
        <td>timeout</td>
        <td>timeout</td>
        <td>timeout</td>
    </tr>
    <tr>
        <td>ABL</td>
        <td><span style="font-weight:bold">94.0</span></td>
        <td><span style="font-weight:bold">89.7</span></td>
        <td><span style="font-weight:bold">96.5</span></td>
        <td><span style="font-weight:bold">97.2</span></td>
        <td><span style="font-weight:bold">98.6</span></td>
        <td><span style="font-weight:bold">77.3</span></td>
    </tr>
    </tbody>
    </table>
    <p style="font-size: 13px;">* timeout: need more than 1 hour to execute</p>