Quick Start
==================

We use the MNIST Add benchmark as a quick start example.

.. code:: python

    import torch
    import torch.nn as nn

    from abl.bridge import SimpleBridge
    from abl.evaluation import SemanticsMetric, SymbolMetric
    from abl.learning import ABLModel, BasicNN
    from abl.reasoning import KBBase, ReasonerBase
    from abl.utils import print_log
    from examples.mnist_add.datasets.get_mnist_add import get_mnist_add
    from examples.models.nn import LeNet5

    # Build logger
    print_log("Abductive Learning on the MNIST Add example.", logger="current")

    # Machine Learning Part
    # Build necessary components for BasicNN
    cls = LeNet5(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cls.parameters(), lr=0.001, betas=(0.9, 0.99))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Build BasicNN
    base_model = BasicNN(cls, criterion, optimizer, device)
    # Build ABLModel
    model = ABLModel(base_model)

    # Logic Part
    # Build knowledge base and reasoner
    class AddKB(KBBase):
        def __init__(self, pseudo_label_list):
            super().__init__(pseudo_label_list)

        # Implement the deduction function
        def logic_forward(self, nums):
            return sum(nums)


    kb = AddKB(pseudo_label_list=list(range(10)))
    reasoner = ReasonerBase(kb, dist_func="confidence")    

    # Datasets and Evaluation Metrics
    # Get training and testing data
    train_data = get_mnist_add(train=True, get_pseudo_label=True)
    test_data = get_mnist_add(train=False, get_pseudo_label=True)
    # Set up metrics
    metric_list = [SymbolMetric(prefix="mnist_add"), SemanticsMetric(kb=kb, prefix="mnist_add")]

    # Bridge Machine Learning and Logic Reasoning
    bridge = SimpleBridge(model, reasoner, metric_list)

    # Train and Test
    bridge.train(train_data, loops=5, segment_size=10000)
    bridge.test(test_data)


Training log would be similar to this:

.. code:: text

    2023/11/29 23:14:17 - abl - INFO - Abductive Learning on the MNIST Add example.
    2023/11/29 23:14:42 - abl - INFO - loop(train) [1/5] segment(train) [1/3] model loss is 1.86793
    2023/11/29 23:14:44 - abl - INFO - loop(train) [1/5] segment(train) [2/3] model loss is 1.48877
    2023/11/29 23:14:46 - abl - INFO - loop(train) [1/5] segment(train) [3/3] model loss is 1.26435
    2023/11/29 23:14:46 - abl - INFO - Evaluation start: loop(val) [1]
    2023/11/29 23:14:47 - abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.334 mnist_add/semantics_accuracy: 0.190 
    2023/11/29 23:14:49 - abl - INFO - loop(train) [2/5] segment(train) [1/3] model loss is 1.06395
    2023/11/29 23:14:51 - abl - INFO - loop(train) [2/5] segment(train) [2/3] model loss is 0.78799
    2023/11/29 23:14:53 - abl - INFO - loop(train) [2/5] segment(train) [3/3] model loss is 0.33641
    2023/11/29 23:14:53 - abl - INFO - Evaluation start: loop(val) [2]
    2023/11/29 23:14:54 - abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.963 mnist_add/semantics_accuracy: 0.926 
    ...
    2023/11/29 23:15:08 - abl - INFO - loop(train) [5/5] segment(train) [1/3] model loss is 0.04223
    2023/11/29 23:15:10 - abl - INFO - loop(train) [5/5] segment(train) [2/3] model loss is 0.03444
    2023/11/29 23:15:12 - abl - INFO - loop(train) [5/5] segment(train) [3/3] model loss is 0.03274
    2023/11/29 23:15:12 - abl - INFO - Evaluation start: loop(val) [5]
    2023/11/29 23:15:13 - abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.991 mnist_add/semantics_accuracy: 0.983 
    2023/11/29 23:15:13 - abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.985 mnist_add/semantics_accuracy: 0.970 
