`Learn the Basics <Basics.html>`_ ||
`Quick Start <Quick-Start.html>`_ ||
`Dataset & Data Structure <Datasets.html>`_ ||
`Learning Part <Learning.html>`_ ||
`Reasoning Part <Reasoning.html>`_ ||
`Evaluation Metrics <Evaluation.html>`_ ||
**Bridge**


Bridge
======

In this section, we will look at how to bridge learning and reasoning parts to train the model, which is the fundamental idea of Abductive Learning. ABL kit implements a set of bridge classes to achieve this.

.. code:: python

    from ablkit.bridge import BaseBridge, SimpleBridge

``BaseBridge`` is an abstract class with the following initialization parameters:

- ``model`` is an object of type ``ABLModel``. The learning part is wrapped in this object.
- ``reasoner`` is an object of type ``Reasoner``. The reasoning part is wrapped in this object.

``BaseBridge`` has the following important methods that need to be overridden in subclasses:

+---------------------------------------+----------------------------------------------------+
| Method Signature                      | Description                                        |
+=======================================+====================================================+
| ``predict(data_examples)``            | Predicts class probabilities and indices           |
|                                       | for the given data examples.                       |
+---------------------------------------+----------------------------------------------------+
| ``abduce_pseudo_label(data_examples)``| Abduces pseudo-labels for the given data examples. |
+---------------------------------------+----------------------------------------------------+
| ``idx_to_pseudo_label(data_examples)``| Converts indices to pseudo-labels using            |
|                                       | the provided or default mapping.                   |
+---------------------------------------+----------------------------------------------------+
| ``pseudo_label_to_idx(data_examples)``| Converts pseudo-labels to indices                  |
|                                       | using the provided or default remapping.           |
+---------------------------------------+----------------------------------------------------+
| ``train(train_data)``                 | Train the model.                                   |
+---------------------------------------+----------------------------------------------------+
| ``test(test_data)``                   | Test the model.                                    |
+---------------------------------------+----------------------------------------------------+

where ``train_data`` and ``test_data`` are both in the form of a tuple or a `ListData <../API/ablkit.data.html#structures.ListData>`_. Regardless of the form, they all need to include three components: ``X``, ``gt_pseudo_label`` and ``Y``. Since ``ListData`` is the underlying data structure used throughout the ABL kit, tuple-formed data will be firstly transformed into ``ListData`` in the ``train`` and ``test`` methods, and such ``ListData`` instances are referred to as ``data_examples``. More details can be found in `preparing datasets <Datasets.html>`_.

``SimpleBridge`` inherits from ``BaseBridge`` and provides a basic implementation. Besides the ``model`` and ``reasoner``, ``SimpleBridge`` has an extra initialization argument, ``metric_list``, which will be used to evaluate model performance. Its training process involves several Abductive Learning loops and each loop consists of the following five steps:

  1. Predict class probabilities and indices for the given data examples.
  2. Transform indices into pseudo-labels.
  3. Revise pseudo-labels based on abdutive reasoning.
  4. Transform the revised pseudo-labels to indices.
  5. Train the model.

The fundamental part of the ``train`` method is as follows:

.. code-block:: python

    def train(self, train_data, loops=50, segment_size=10000):
        """
        Parameters
        ----------
        train_data : Union[ListData, Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]]
            Training data should be in the form of ``(X, gt_pseudo_label, Y)`` or a ``ListData``
            object with ``X``, ``gt_pseudo_label`` and ``Y`` attributes.
            - ``X`` is a list of sublists representing the input data.
            - ``gt_pseudo_label`` is only used to evaluate the performance of the ``ABLModel`` but not
            to train. ``gt_pseudo_label`` can be ``None``.
            - ``Y`` is a list representing the ground truth reasoning result for each sublist in ``X``.
        loops : int
            Learning part and Reasoning part will be iteratively optimized for ``loops`` times.
        segment_size : Union[int, float]
            Data will be split into segments of this size and data in each segment
            will be used together to train the model.
        """
        if isinstance(train_data, ListData):
            data_examples = train_data
        else:
            data_examples = self.data_preprocess(*train_data)
        
        if isinstance(segment_size, float):
            segment_size = int(segment_size * len(data_examples))

        for loop in range(loops):
            for seg_idx in range((len(data_examples) - 1) // segment_size + 1):
                sub_data_examples = data_examples[
                    seg_idx * segment_size : (seg_idx + 1) * segment_size
                ]
                self.predict(sub_data_examples)                  # 1
                self.idx_to_pseudo_label(sub_data_examples)      # 2
                self.abduce_pseudo_label(sub_data_examples)      # 3
                self.pseudo_label_to_idx(sub_data_examples)      # 4
                loss = self.model.train(sub_data_examples)       # 5, self.model is an ABLModel object

