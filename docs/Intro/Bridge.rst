`Learn the Basics <Basics.html>`_ ||
`Quick Start <Quick-Start.html>`_ ||
`Dataset & Data Structure <Datasets.html>`_ ||
`Learning Part <Learning.html>`_ ||
`Reasoning Part <Reasoning.html>`_ ||
`Evaluation Metrics <Evaluation.html>`_ ||
**Bridge**


Bridge
======

Bridging machine learning and reasoning to train the model is the fundamental idea of Abductive Learning, ABL-Package implements a set of `bridge class <../API/abl.bridge.html>`_ to achieve this.

``BaseBridge`` is an abstract class with the following initialization parameters:

- ``model``: an object of type ``ABLModel``. Machine Learning part are wrapped in this object.
- ``reasoner``: a object of type ``Reasoner``. Reasoning part are wrapped in this object.

``BaseBridge`` has the following important methods that need to be overridden in subclasses:

+-----------------------------------+--------------------------------------------------------------------------------------+
| Method Signature                  | Description                                                                          |
+===================================+======================================================================================+
| predict(data_samples)             | Predicts class probabilities and indices for the given data samples.                 |
+-----------------------------------+--------------------------------------------------------------------------------------+
| abduce_pseudo_label(data_samples) | Abduces pseudo labels for the given data samples.                                    |
+-----------------------------------+--------------------------------------------------------------------------------------+
| idx_to_pseudo_label(data_samples) | Converts indices to pseudo labels using the provided or default mapping.             |
+-----------------------------------+--------------------------------------------------------------------------------------+
| pseudo_label_to_idx(data_samples) | Converts pseudo labels to indices using the provided or default remapping.           |
+-----------------------------------+--------------------------------------------------------------------------------------+
| train(train_data)                 | Train the model.                                                                     |
+-----------------------------------+--------------------------------------------------------------------------------------+
| test(test_data)                   | Test the model.                                                                      |
+-----------------------------------+--------------------------------------------------------------------------------------+

where ``train_data`` and ``test_data`` are both in the form of ``(X, gt_pseudo_label, Y)``. They will be used to construct ``ListData`` instances which are referred to as ``data_samples`` in the ``train`` and ``test`` methods respectively. More details can be found in `preparing datasets <Datasets.html>`_.


``SimpleBridge`` inherits from ``BaseBridge`` and provides a basic implementation. Besides the ``model`` and ``reasoner``, ``SimpleBridge`` has an extra initialization arguments, ``metric_list``, which will be used to evaluate model performance. Its training process involves several Abductive Learning loops and each loop consists of the following five steps:

  1. Predict class probabilities and indices for the given data samples.
  2. Transform indices into pseudo labels.
  3. Revise pseudo labels based on abdutive reasoning.
  4. Transform the revised pseudo labels to indices.
  5. Train the model.

The fundamental part of the ``train`` method is as follows:

.. code-block:: python

    def train(self, train_data, loops=50, segment_size=10000):
        """
        Parameters
        ----------
        train_data : Tuple[List[List[Any]], Optional[List[List[Any]]], List[Any]]]
            Training data.
        loops : int
            Machine Learning part and Reasoning part will be iteratively optimized
            for ``loops`` times.
        segment_size : Union[int, float]
            Data will be split into segments of this size and data in each segment
            will be used together to train the model.
        """
        if isinstance(train_data, ListData):
            data_samples = train_data
        else:
            data_samples = self.data_preprocess(*train_data)
        
        if isinstance(segment_size, float):
            segment_size = int(segment_size * len(data_samples))

        for loop in range(loops):
            for seg_idx in range((len(data_samples) - 1) // segment_size + 1):
                sub_data_samples = data_samples[
                    seg_idx * segment_size : (seg_idx + 1) * segment_size
                ]
                self.predict(sub_data_samples)                  # 1
                self.idx_to_pseudo_label(sub_data_samples)      # 2
                self.abduce_pseudo_label(sub_data_samples)      # 3
                self.pseudo_label_to_idx(sub_data_samples)      # 4
                loss = self.model.train(sub_data_samples)       # 5, self.model is an ABLModel object

