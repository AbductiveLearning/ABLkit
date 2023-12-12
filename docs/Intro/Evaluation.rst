`Learn the Basics <Basics.html>`_ ||
`Quick Start <Quick-Start.html>`_ ||
`Dataset & Data Structure <Datasets.html>`_ ||
`Learning Part <Learning.html>`_ ||
`Reasoning Part <Reasoning.html>`_ ||
**Evaluation Metrics** ||
`Bridge <Bridge.html>`_


Evaluation Metrics
==================

ABL-Package seperates the evaluation process from model training and testing as an independent class, ``BaseMetric``. The training and testing processes are implemented in the ``BaseBridge`` class, so metrics are used by this class and its sub-classes. After building a ``bridge`` with a list of ``BaseMetric`` instances, these metrics will be used by the ``bridge.valid`` method to evaluate the model performance during training and testing. 

To customize our own metrics, we need to inherit from ``BaseMetric`` and implement the ``process`` and ``compute_metrics`` methods. 

- The ``process`` method accepts a batch of model prediction and saves the information to ``self.results`` property after processing this batch. 
- The ``compute_metrics`` method uses all the information saved in ``self.results`` to calculate and return a dict that holds the evaluation results. 

Besides, we can assign a ``str`` to the ``prefix`` argument of the ``__init__`` method. This string is automatically prefixed to the output metric names. For example, if we set ``prefix="mnist_add"``, the output metric name will be ``character_accuracy``.
We provide two basic metrics, namely ``SymbolMetric`` and ``ReasoningMetric``, which are used to evaluate the accuracy of the machine learning model's predictions and the accuracy of the ``logic_forward`` results, respectively. Using ``SymbolMetric`` as an example, the following code shows how to implement a custom metrics.

.. code:: python

    class SymbolMetric(BaseMetric):
        def __init__(self, prefix: Optional[str] = None) -> None:
            # prefix is used to distinguish different metrics
            super().__init__(prefix)

        def process(self, data_samples: Sequence[dict]) -> None:
            # pred_pseudo_label and gt_pseudo_label are both of type List[List[Any]] 
            # and have the same length
            pred_pseudo_label = data_samples.pred_pseudo_label
            gt_pseudo_label = data_samples.gt_pseudo_label
            
            for pred_z, z in zip(pred_pseudo_label, gt_pseudo_label):
                correct_num = 0
                for pred_symbol, symbol in zip(pred_z, z):
                    if pred_symbol == symbol:
                        correct_num += 1
                self.results.append(correct_num / len(z))
        
        def compute_metrics(self, results: list) -> dict:
            metrics = dict()
            metrics["character_accuracy"] = sum(results) / len(results)
            return metrics