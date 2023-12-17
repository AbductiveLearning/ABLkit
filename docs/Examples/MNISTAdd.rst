MNIST Addition
==============

This example shows a simple implementation of MNIST Addition, which was
first introduced in `Manhaeve et al.,
2018 <https://arxiv.org/abs/1805.10872>`__. In this task, the inputs are
pairs of MNIST handwritten images, and the outputs are their sums.

In Abductive Learning, we hope to first use learning part to map the
input images to their digits (we call it pseudo labels), and then use
reasoning part to calculate the summation of these pseudo labels to get
the final result.

.. code:: ipython3

    import os.path as osp
    
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    
    from abl.bridge import SimpleBridge
    from abl.evaluation import ReasoningMetric, SymbolMetric
    from abl.learning import ABLModel, BasicNN
    from abl.reasoning import KBBase, Reasoner
    from abl.utils import ABLLogger, print_log
    from examples.mnist_add.datasets import get_mnist_add
    from examples.models.nn import LeNet5

Load Datasets
-------------

First, we get training and testing data:

.. code:: ipython3

    train_data = get_mnist_add(train=True, get_pseudo_label=True)
    test_data = get_mnist_add(train=False, get_pseudo_label=True)

The datasets are illustrated as follows:

.. code:: ipython3

    print(f"There are {len(train_data[0])} data examples in the training set and {len(test_data[0])} data examples in the test set")
    print(f"Each of the data example has {len(train_data)} components: X, gt_pseudo_label, and Y.")
    print("As an illustration, in the First data example of the training set, we have:")
    print(f"X ({len(train_data[0][0])} images):")
    plt.subplot(1,2,1)
    plt.axis('off') 
    plt.imshow(train_data[0][0][0].numpy().transpose(1, 2, 0))
    plt.subplot(1,2,2)
    plt.axis('off') 
    plt.imshow(train_data[0][0][1].numpy().transpose(1, 2, 0))
    plt.show()
    print(f"gt_pseudo_label ({len(train_data[1][0])} ground truth pseudo label): {train_data[1][0][0]}, {train_data[1][0][1]}")
    print(f"Y (their sum result): {train_data[2][0]}")

Out:
   .. code:: none
      :class: code-out

      There are 30000 data examples in the training set and 5000 data examples in the test set
      Each of the data example has 3 components: X, gt_pseudo_label, and Y.
      As an illustration, in the First data example of the training set, we have:
      X (2 images): 

   .. image:: ../img/mnist_add_datasets.png
      :width: 400px

   .. code:: none
      :class: code-out

      gt_pseudo_label (2 ground truth pseudo label): 7, 5
      Y (their sum result): 12



Learning Part
-------------

First, we build the basic learning model. We use a simple `LeNet neural
network <https://en.wikipedia.org/wiki/LeNet>`__ to complete this task.

.. code:: ipython3

    cls = LeNet5(num_classes=10)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cls.parameters(), lr=0.001, betas=(0.9, 0.99))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    base_model = BasicNN(
        cls,
        loss_fn,
        optimizer,
        device,
        batch_size=32,
        num_epochs=1,
    )

The base model can predict the outcome class index and the probabilities
for an image, as shown below:

.. code:: ipython3

    pred_idx = base_model.predict(X=[torch.randn(1, 28, 28).to(device) for _ in range(32)])
    print(f"Shape of pred_idx for a batch of 32 samples: {pred_idx.shape}")
    pred_prob = base_model.predict_proba(X=[torch.randn(1, 28, 28).to(device) for _ in range(32)])
    print(f"Shape of pred_prob for a batch of 32 samples: {pred_prob.shape}")

Out:
   .. code:: none
      :class: code-out

      Shape of pred_idx for a batch of 32 samples: (32,)
      Shape of pred_prob for a batch of 32 samples: (32, 10)
    

Then, we build an instance of ``ABLModel``. The main function of
``ABLModel`` is to serialize data and provide a unified interface for
different base machine learning models.

.. code:: ipython3

    model = ABLModel(base_model)

Logic Part
----------

In the logic part, we first build a knowledge base.

.. code:: ipython3

    # Build knowledge base and reasoner
    class AddKB(KBBase):
        def __init__(self, pseudo_label_list):
            super().__init__(pseudo_label_list)
    
        # Implement the deduction function
        def logic_forward(self, nums):
            return sum(nums)
    
    kb = AddKB(pseudo_label_list=list(range(10)))

The knowledge base can perform logical reasoning. Below is an example of
performing (deductive) reasoning:

.. code:: ipython3

    pseudo_label_sample = [1, 2]
    reasoning_result = kb.logic_forward(pseudo_label_sample)
    print(f"Reasoning result of pseudo label sample {pseudo_label_sample} is {reasoning_result}.")

Out:
   .. code:: none
      :class: code-out

      Reasoning result of pseudo label sample [1, 2] is 3.
    

Then, we create a reasoner. It can help minimize inconsistencies between
the knowledge base and pseudo labels predicted by the learning part.

.. code:: ipython3

    reasoner = Reasoner(kb, dist_func="confidence")

Evaluation Metrics
------------------

Set up evaluation metrics. These metrics will be used to evaluate the
model performance during training and testing.

.. code:: ipython3

    metric_list = [SymbolMetric(prefix="mnist_add"), ReasoningMetric(kb=kb, prefix="mnist_add")]

Bridge Learning and Reasoning
-----------------------------

Now, the last step is to bridge the learning and reasoning part.

.. code:: ipython3

    bridge = SimpleBridge(model, reasoner, metric_list)

Perform training and testing.

.. code:: ipython3

    # Build logger
    print_log("Abductive Learning on the MNIST Addition example.", logger="current")
    
    # Retrieve the directory of the Log file and define the directory for saving the model weights.
    log_dir = ABLLogger.get_current_instance().log_dir
    weights_dir = osp.join(log_dir, "weights")

    bridge.train(train_data, loops=5, segment_size=1/3, save_interval=1, save_dir=weights_dir)
    bridge.test(test_data)

Out:
   .. code:: none
      :class: code-out

      abl - INFO - Abductive Learning on the MNIST Addition example.
      abl - INFO - loop(train) [1/5] segment(train) [1/3] 
      abl - INFO - model loss: 1.81231
      abl - INFO - loop(train) [1/5] segment(train) [2/3] 
      abl - INFO - model loss: 1.37639
      abl - INFO - loop(train) [1/5] segment(train) [3/3] 
      abl - INFO - model loss: 1.14446
      abl - INFO - Evaluation start: loop(val) [1]
      abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.207 mnist_add/reasoning_accuracy: 0.245 
      abl - INFO - Saving model: loop(save) [1]
      abl - INFO - Checkpoints will be saved to results/20231217_14_27_56/weights/model_checkpoint_loop_1.pth
      abl - INFO - loop(train) [2/5] segment(train) [1/3] 
      abl - INFO - model loss: 0.97430
      abl - INFO - loop(train) [2/5] segment(train) [2/3] 
      abl - INFO - model loss: 0.91448
      abl - INFO - loop(train) [2/5] segment(train) [3/3] 
      abl - INFO - model loss: 0.83089
      abl - INFO - Evaluation start: loop(val) [2]
      abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.191 mnist_add/reasoning_accuracy: 0.353 
      abl - INFO - Saving model: loop(save) [2]
      abl - INFO - Checkpoints will be saved to results/20231217_14_27_56/weights/model_checkpoint_loop_2.pth
      abl - INFO - loop(train) [3/5] segment(train) [1/3] 
      abl - INFO - model loss: 0.79906
      abl - INFO - loop(train) [3/5] segment(train) [2/3] 
      abl - INFO - model loss: 0.77949
      abl - INFO - loop(train) [3/5] segment(train) [3/3] 
      abl - INFO - model loss: 0.75007
      abl - INFO - Evaluation start: loop(val) [3]
      abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.148 mnist_add/reasoning_accuracy: 0.385 
      abl - INFO - Saving model: loop(save) [3]
      abl - INFO - Checkpoints will be saved to results/20231217_14_27_56/weights/model_checkpoint_loop_3.pth
      abl - INFO - loop(train) [4/5] segment(train) [1/3] 
      abl - INFO - model loss: 0.72659
      abl - INFO - loop(train) [4/5] segment(train) [2/3] 
      abl - INFO - model loss: 0.70985
      abl - INFO - loop(train) [4/5] segment(train) [3/3] 
      abl - INFO - model loss: 0.66337
      abl - INFO - Evaluation start: loop(val) [4]
      abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.016 mnist_add/reasoning_accuracy: 0.494 
      abl - INFO - Saving model: loop(save) [4]
      abl - INFO - Checkpoints will be saved to results/20231217_14_27_56/weights/model_checkpoint_loop_4.pth
      abl - INFO - loop(train) [5/5] segment(train) [1/3] 
      abl - INFO - model loss: 0.61140
      abl - INFO - loop(train) [5/5] segment(train) [2/3] 
      abl - INFO - model loss: 0.57534
      abl - INFO - loop(train) [5/5] segment(train) [3/3] 
      abl - INFO - model loss: 0.57018
      abl - INFO - Evaluation start: loop(val) [5]
      abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.002 mnist_add/reasoning_accuracy: 0.507 
      abl - INFO - Saving model: loop(save) [5]
      abl - INFO - Checkpoints will be saved to results/20231217_14_27_56/weights/model_checkpoint_loop_5.pth
      abl - INFO - Evaluation ended, mnist_add/character_accuracy: 0.002 mnist_add/reasoning_accuracy: 0.482 
      
More concrete examples are available in `examples/mnist_add` folder.