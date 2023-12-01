Define Evaluation Metrics
=========================

To validate and test the model, we need to inherit from ``BaseMetric`` to define metrics and implement the ``process`` and ``compute_metrics`` methods where the process method accepts a batch of outputs. After processing this batch of data, we save the information to ``self.results`` property. The input results of ``compute_metrics`` is all the information saved in ``process``. Use these information to calculate and return a dict that holds the results of the evaluation metrics. 

We provide two basic metrics, namely ``SymbolMetric`` and ``SemanticsMetric``, which are used to evaluate the accuracy of the machine learning model's predictions and the accuracy of the ``logic_forward`` results, respectively.

In the case of MNIST Add example, the metric definition looks like

.. code:: python

    metric_list = [SymbolMetric(prefix="mnist_add"), SemanticsMetric(kb=kb, prefix="mnist_add")]