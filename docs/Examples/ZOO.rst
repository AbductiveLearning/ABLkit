ZOO
===

Below shows an implementation of
`Zoo <https://archive.ics.uci.edu/dataset/111/zoo>`__. In this task,
attributes of animals (such as presence of hair, eggs, etc.) and their
targets (the animal class they belong to) are given, along with a
knowledge base which contain information about the relations between
attributes and targets, e.g., Implies(milk == 1, mammal == 1).

The goal of this task is to develop a learning model that can predict
the targets of animals based on their attributes. In the initial stages,
when the model is under-trained, it may produce incorrect predictions
that conflict with the relations contained in the knowledge base. When
this happens, abductive reasoning can be employed to adjust these
results and retrain the model accordingly. This process enables us to
further update the learning model.

.. code:: ipython3

    # Import necessary libraries and modules
    import os.path as osp
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from examples.zoo.get_dataset import load_and_preprocess_dataset, split_dataset
    from abl.learning import ABLModel
    from examples.zoo.kb import ZooKB
    from abl.reasoning import Reasoner
    from abl.data.evaluation import ReasoningMetric, SymbolAccuracy
    from abl.utils import ABLLogger, print_log, confidence_dist
    from abl.bridge import SimpleBridge

Working with Data
-----------------

First, we load and preprocess the `Zoo
dataset <https://archive.ics.uci.edu/dataset/111/zoo>`__, and split it
into labeled/unlabeled/test data

.. code:: ipython3

    X, y = load_and_preprocess_dataset(dataset_id=62)
    X_label, y_label, X_unlabel, y_unlabel, X_test, y_test = split_dataset(X, y, test_size=0.3)

Zoo dataset consist of tabular data. The attributes contains 17 boolean
values (e.g., hair, feathers, eggs, milk, airborne, aquatic, etc.) and
the target is a integer value in range [0,6] representing 7 classes
(e.g., mammal, bird, reptile, fish, amphibian, insect, and other). Below
is an illustration:

.. code:: ipython3

    print("Shape of X and y:", X.shape, y.shape)
    print("First five elements of X:")
    print(X[:5])
    print("First five elements of y:")
    print(y[:5])

Out:
    .. code:: none
        :class: code-out

        Shape of X and y: (101, 16) (101,)
        First five elements of X:
        [[True False False True False False True True True True False False 4
        False False True]
        [True False False True False False False True True True False False 4
        True False True]
        [False False True False False True True True True False False True 0
        True False False]
        [True False False True False False True True True True False False 4
        False False True]
        [True False False True False False True True True True False False 4
        True False True]]
        First five elements of y:
        [0 0 3 0 0]
    

Next, we transform the tabular data to the format required by
ABL-Package, which is a tuple of (X, gt_pseudo_label, Y). In this task,
we treat the attributes as X and the targets as gt_pseudo_label (ground
truth pseudo-labels). Y (reasoning results) are expected to be 0,
indicating no rules are violated.

.. code:: ipython3

    def transform_tab_data(X, y):
        return ([[x] for x in X], [[y_item] for y_item in y], [0] * len(y))
    label_data = transform_tab_data(X_label, y_label)
    test_data = transform_tab_data(X_test, y_test)
    train_data = transform_tab_data(X_unlabel, y_unlabel)

Building the Learning Part
--------------------------

To build the learning part, we need to first build a machine learning
base model. We use a `Random
Forest <https://en.wikipedia.org/wiki/Random_forest>`__ as the base
model.

.. code:: ipython3

    base_model = RandomForestClassifier()

However, the base model built above deals with instance-level data, and
can not directly deal with example-level data. Therefore, we wrap the
base model into ``ABLModel``, which enables the learning part to train,
test, and predict on example-level data.

.. code:: ipython3

    model = ABLModel(base_model)

Building the Reasoning Part
---------------------------

In the reasoning part, we first build a knowledge base which contains
information about the relations between attributes (X) and targets
(pseudo-labels), e.g., Implies(milk == 1, mammal == 1). The knowledge
base is built in the ``ZooKB`` class within file ``examples/zoo/kb.py``, and is
derived from the ``KBBase`` class.

.. code:: ipython3

    kb = ZooKB()

As mentioned, for all attributes and targets in the dataset, the
reasoning results are expected to be 0 since there should be no
violations of the established knowledge in real data. As shown below:

.. code:: ipython3

    for idx, (x, y_item) in enumerate(zip(X[:5], y[:5])):
        print(f"Example {idx}: the attributes are: {x}, and the target is {y_item}.")
        print(f"Reasoning result is {kb.logic_forward([y_item], [x])}.")
        print()

Out:
    .. code:: none
        :class: code-out

        Example 0: the attributes are: [True False False True False False True True True True False False 4 False
        False True], and the target is 0.
        Reasoning result is 0.
        
        Example 1: the attributes are: [True False False True False False False True True True False False 4 True
        False True], and the target is 0.
        Reasoning result is 0.
        
        Example 2: the attributes are: [False False True False False True True True True False False True 0 True
        False False], and the target is 3.
        Reasoning result is 0.
        
        Example 3: the attributes are: [True False False True False False True True True True False False 4 False
        False True], and the target is 0.
        Reasoning result is 0.
        
        Example 4: the attributes are: [True False False True False False True True True True False False 4 True
        False True], and the target is 0.
        Reasoning result is 0.
    
    

Then, we create a reasoner by instantiating the class ``Reasoner``. Due
to the indeterminism of abductive reasoning, there could be multiple
candidates compatible to the knowledge base. When this happens, reasoner
can minimize inconsistencies between the knowledge base and
pseudo-labels predicted by the learning part, and then return only one
candidate that has the highest consistency.

.. code:: ipython3

    def consitency(data_example, candidates, candidate_idxs, reasoning_results):
        pred_prob = data_example.pred_prob
        model_scores = confidence_dist(pred_prob, candidate_idxs)
        rule_scores = np.array(reasoning_results)
        scores = model_scores + rule_scores
        return scores
    
    reasoner = Reasoner(kb, dist_func=consitency)

Building Evaluation Metrics
---------------------------

Next, we set up evaluation metrics. These metrics will be used to
evaluate the model performance during training and testing.
Specifically, we use ``SymbolAccuracy`` and ``ReasoningMetric``, which
are used to evaluate the accuracy of the machine learning model’s
predictions and the accuracy of the final reasoning results,
respectively.

.. code:: ipython3

    metric_list = [SymbolAccuracy(prefix="zoo"), ReasoningMetric(kb=kb, prefix="zoo")]

Bridging Learning and Reasoning
-------------------------------

Now, the last step is to bridge the learning and reasoning part. We
proceed this step by creating an instance of ``SimpleBridge``.

.. code:: ipython3

    bridge = SimpleBridge(model, reasoner, metric_list)

Perform training and testing by invoking the ``train`` and ``test``
methods of ``SimpleBridge``.

.. code:: ipython3

    # Build logger
    print_log("Abductive Learning on the ZOO example.", logger="current")
    log_dir = ABLLogger.get_current_instance().log_dir
    weights_dir = osp.join(log_dir, "weights")
    
    print_log("------- Use labeled data to pretrain the model -----------", logger="current")
    base_model.fit(X_label, y_label)
    print_log("------- Test the initial model -----------", logger="current")
    bridge.test(test_data)
    print_log("------- Use ABL to train the model -----------", logger="current")
    bridge.train(train_data=train_data, label_data=label_data, loops=3, segment_size=len(X_unlabel), save_dir=weights_dir)
    print_log("------- Test the final model -----------", logger="current")
    bridge.test(test_data)


Out:
    .. code:: none
        :class: code-out

        abl - INFO - Abductive Learning on the ZOO example.
        abl - INFO - ------- Use labeled data to pretrain the model -----------
        abl - INFO - ------- Test the initial model -----------
        abl - INFO - Evaluation ended, zoo/character_accuracy: 0.935 zoo/reasoning_accuracy: 0.935 
        abl - INFO - ------- Use ABL to train the model -----------
        abl - INFO - loop(train) [1/3] segment(train) [1/1] 
        abl - INFO - Evaluation start: loop(val) [1]
        abl - INFO - Evaluation ended, zoo/character_accuracy: 0.984 zoo/reasoning_accuracy: 1.000 
        abl - INFO - loop(train) [2/3] segment(train) [1/1] 
        abl - INFO - Evaluation start: loop(val) [2]
        abl - INFO - Evaluation ended, zoo/character_accuracy: 0.984 zoo/reasoning_accuracy: 1.000 
        abl - INFO - loop(train) [3/3] segment(train) [1/1] 
        abl - INFO - Evaluation start: loop(val) [3]
        abl - INFO - Evaluation ended, zoo/character_accuracy: 0.984 zoo/reasoning_accuracy: 1.000 
        abl - INFO - ------- Test the final model -----------
        abl - INFO - Evaluation ended, zoo/character_accuracy: 0.903 zoo/reasoning_accuracy: 0.935 
        

We may see from the results, after undergoing training with ABL, the
model’s accuracy has improved.

More concrete examples are available in ``examples/zoo/main.py`` and ``examples/zoo/zoo.ipynb``.
