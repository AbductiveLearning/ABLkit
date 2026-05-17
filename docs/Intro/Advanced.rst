.. _Advanced:

Advanced Topics
===============

The standard ABL pipeline (``BasicNN`` + ``ABLModel`` + ``Reasoner`` +
``SimpleBridge``) covers the majority of tasks. ABLkit also ships a few
drop-in variants for settings where the standard pipeline does not quite
fit. This page collects them in one place.

The four topics below are independent: pick whichever ones apply to your
task.

* :ref:`advanced-multilabel`: when each instance can carry **multiple
  active labels** (sigmoid + binary indicator vectors) rather than a
  single class.
* :ref:`advanced-semisupervised`: when **part of the training set
  carries ground-truth pseudo-labels** and the rest must be abduced.
* :ref:`advanced-a3bl`: when many label assignments are consistent with
  the knowledge base and we want to aggregate them into a **soft
  label** instead of picking the single best one.
* :ref:`advanced-verification`: when we want to train against the
  **top-K consistent label assignments** by joint probability rather
  than a single best candidate.


.. _advanced-multilabel:

Multi-Label Models
------------------

By default ``BasicNN`` and ``ABLModel`` assume a single-label
multi-class setting: softmax over classes, ``argmax`` at prediction
time, one integer label per instance. For tasks where each instance is
described by a *vector* of independent binary attributes (e.g., the 21
binary concepts in BDD-OIA), ABLkit provides multi-label drop-in
replacements:

* :class:`~ablkit.learning.MultiLabelBasicNN`: sigmoid output,
  threshold at 0.5 for prediction, ``MultiLabelClassificationDataset``
  for training.
* :class:`~ablkit.learning.MultiLabelABLModel`: wraps a multi-label
  base model and thresholds per-label probabilities into binary
  indicator vectors stored on ``pred_idx``.
* :class:`~ablkit.learning.MultiLabelClassificationDataset`: stores
  ``Y`` as a ``FloatTensor`` so it can be fed directly into
  ``BCEWithLogitsLoss``.

Typical usage swaps the standard classes 1-for-1:

.. code:: python

    import torch.nn as nn
    from torch import optim

    from ablkit.learning import MultiLabelABLModel, MultiLabelBasicNN

    net = MyMultiLabelNet()          # PyTorch model with num_labels outputs
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=2e-3)

    base_model = MultiLabelBasicNN(net, loss_fn, optimizer, device="cpu",
                                   batch_size=32, num_epochs=1)
    model = MultiLabelABLModel(base_model)

See the BDD-OIA example for an end-to-end multi-label pipeline.


.. _advanced-semisupervised:

Semi-Supervised Training
------------------------

When part of the training set already carries ground-truth
pseudo-labels (and the rest is unlabeled), ``SimpleBridge`` can be
asked to use those labels directly instead of abducing them.

The mechanism is purely a flag on ``SimpleBridge.train``:

* Provide a ``train_data`` tuple ``(X, gt_pseudo_label, Y)`` where the
  ``gt_pseudo_label`` for unlabeled examples is ``None``.
* Pass ``use_supervised_data=True``.

Under the hood the bridge calls
``Reasoner.batch_supervised_abduce``, which keeps existing
``gt_pseudo_label`` values verbatim and only abduces a candidate for
the ``None`` entries:

.. code:: python

    bridge.train(
        train_data=(X, pseudo_label_with_some_None, Y),
        use_supervised_data=True,
        loops=50,
        segment_size=0.01,
    )

The ``--labeled-ratio`` flag in the MNIST Addition example
demonstrates how to mask out a fraction of pseudo-labels and feed the
result through this flow.


.. _advanced-a3bl:

A3BL: Ambiguity-Aware Abductive Learning
----------------------------------------

When many label assignments are consistent with the knowledge base
for a given example, picking only the lowest-distance candidate
discards useful signal. A3BL (Ambiguity-Aware Abductive Learning)
keeps the top candidates, weights them by their joint probability,
and trains the model on the resulting *soft label distribution*.

ABLkit ships two classes:

* :class:`~ablkit.reasoning.A3BLReasoner`: enumerates valid
  candidates, scores them via a softmax over per-symbol probabilities,
  and aggregates the top-K into a soft label.
* :class:`~ablkit.bridge.A3BLBridge`: runs the ambiguity-aware
  prediction → soft-label-abduction → train loop.

Minimal wiring:

.. code:: python

    from ablkit.bridge import A3BLBridge
    from ablkit.reasoning import A3BLReasoner

    reasoner = A3BLReasoner(kb, topK=16, temperature=0.2)
    bridge = A3BLBridge(model, reasoner, metric_list)
    bridge.train(train_data, loops=2, segment_size=0.01)

Reference: https://github.com/Hao-Yuan-He/A3BL


.. _advanced-verification:

Verification Learning
---------------------

Verification Learning replaces the standard "abduce the single best
candidate" step with a top-K enumeration: starting from the most
probable joint label assignment, the search walks the per-symbol
probability lattice in **descending joint-probability order** and
collects the first ``top_k`` candidates that satisfy the knowledge
base. The model is then trained once per candidate per segment.

ABLkit ships two classes (consolidated in
``ablkit/reasoning/reasoner.py`` and
``ablkit/bridge/verification_bridge.py``):

* :class:`~ablkit.reasoning.VerificationReasoner`: exposes
  ``top_k_candidates(pred_prob, y)`` and the batched variant.
* :class:`~ablkit.bridge.VerificationBridge`: drives the
  predict → enumerate → train-per-candidate loop.

Helpers usable without a reasoner instance:

* :func:`ablkit.reasoning.reasoner.enumerate_label_assignments`: a
  generator over label assignments in descending joint-probability
  order.
* :func:`ablkit.reasoning.reasoner.top_k_satisfying`: wraps the
  generator with a user predicate and a fallback when nothing matches.

Minimal wiring:

.. code:: python

    from ablkit.bridge import VerificationBridge
    from ablkit.reasoning import VerificationReasoner

    reasoner = VerificationReasoner(kb, top_k=3, max_iter=10000)
    bridge = VerificationBridge(model, reasoner, metric_list)
    bridge.train(train_data, loops=2, segment_size=0.01)

Reference: https://github.com/VerificationLearning/VerificationLearning

The ``--method verification --top-k K`` flags in the MNIST Addition
example demonstrate the full pipeline.
