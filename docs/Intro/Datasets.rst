Prepare datasets
================

Next, we need to build datasets. ABL-Package assumes data to be in the form of ``(X, gt_pseudo_label, Y)`` where ``X`` is the input of the machine learning model, ``Y`` is the ground truth of the reasoning result and ``gt_pseudo_label`` is the ground truth label of each element in ``X``. ``X`` should be of type ``List[List[Any]]``, ``Y`` should be of type ``List[Any]`` and ``gt_pseudo_label`` can be ``None`` or of the type ``List[List[Any]]``. 

In the MNIST Add example, the data loading looks like

.. code:: python

    # train_data and test_data are all tuples consist of X, gt_pseudo_label and Y.
    train_data = get_mnist_add(train=True, get_pseudo_label=True)
    test_data = get_mnist_add(train=False, get_pseudo_label=True)

