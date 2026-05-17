BDD-OIA
=======

.. raw:: html

    <p>For detailed code implementation, please view it on <a class="reference external" href="https://github.com/AbductiveLearning/ABLkit/tree/main/examples/bdd_oia" target="_blank">GitHub</a>.</p>

Below shows an implementation of `BDD-OIA <https://twizwei.github.io/bddoia_project/>`__.
The BDD-OIA dataset comprises frames extracted from driving scene videos
that are used for autonomous driving predictions. Each frame is
annotated with 4 binary action labels (:math:`\textsf{move_forward}`,
:math:`\textsf{stop}`, :math:`\textsf{turn_left}`, :math:`\textsf{turn_right}`),
as well as 21 intermediate binary concept labels such as
:math:`\textsf{red_light}` and :math:`\textsf{road_clear}` that explain those
actions.

The objective is to predict the possible actions for each frame.
During training we use only the action-level supervision together with
a knowledge base that captures the relations between concepts and
actions, e.g.,
:math:`\textsf{red_light} \lor \textsf{traffic_sign} \lor \textsf{obstacle} \implies \textsf{stop}`.
The training set contains 16,000 frames; the test set contains 4,500.

Intuitively, the learning part predicts the 21 binary concept
pseudo-labels from each frame, and the reasoning part uses the
knowledge base to derive the four action labels from those concepts.
When the learning part's predictions conflict with the ground-truth
actions, the reasoner revises the concepts via abductive reasoning,
and those revised concepts are used to further train the learning
part.

The dataset was preprocessed by `Marconato et al. (2023) <https://proceedings.neurips.cc/paper_files/paper/2023/file/e560202b6e779a82478edb46c6f8f4dd-Paper-Conference.pdf>`__
with a pretrained Faster-RCNN on BDD-100k together with the first
module of CBM-AUC `(Sawada & Nakamura, 2022) <https://arxiv.org/abs/2202.01459>`__,
yielding a 2048-dimensional visual feature for each frame.

.. code:: python

    # Import necessary libraries and modules
    import os.path as osp

    import numpy as np
    import torch
    import torch.nn as nn
    from torch import optim

    from ablkit.data.evaluation import SymbolAccuracy
    from ablkit.learning import MultiLabelABLModel, MultiLabelBasicNN
    from ablkit.reasoning import KBBase, Reasoner
    from ablkit.utils import ABLLogger, print_log

    from bridge import BDDBridge
    from dataset.data_util import get_dataset
    from metric import BDDReasoningMetric
    from models.nn import ConceptNet

Working with Data
-----------------

First, we load the training, validation, and testing splits:

.. code:: python

    train_data = get_dataset(fname="train.npz", get_pseudo_label=True)
    val_data = get_dataset(fname="val.npz", get_pseudo_label=True)
    test_data = get_dataset(fname="test.npz", get_pseudo_label=True)

Each split consists of three components (``X``, ``gt_pseudo_label``,
and ``Y``) with one entry per frame:

- ``X[i]`` is a list with a single ndarray of shape ``(2048,)``, the
  pre-extracted visual feature for the frame.
- ``gt_pseudo_label[i]`` is a list of length 21 holding the binary
  concept annotations (``red_light``, ``road_clear``, …).
- ``Y[i]`` is a tuple of length 4 holding the binary action labels
  (``move_forward``, ``stop``, ``turn_left``, ``turn_right``).

During training only ``X`` and ``Y`` are used; ``gt_pseudo_label`` is
held back for evaluation.

Building the Learning Part
--------------------------

To build the learning part we first construct a PyTorch model,
``ConceptNet``, then wrap it in
:class:`~ablkit.learning.MultiLabelBasicNN` to obtain an sklearn-style
base model. ``MultiLabelBasicNN`` is a multi-label variant of
``BasicNN``: the output uses sigmoid activations rather than softmax,
predictions are binary vectors rather than single class indices, and
the dataset is a
:class:`~ablkit.learning.MultiLabelClassificationDataset`. The 21
outputs therefore correspond to the 21 binary concept labels.

.. code:: python

    net = ConceptNet()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.002,
        pct_start=0.15,
        epochs=2,
        steps_per_epoch=int(1 / 0.01) + 1,
    )

    base_model = MultiLabelBasicNN(
        net,
        loss_fn,
        optimizer,
        scheduler=scheduler,
        device=device,
        batch_size=32,
        num_epochs=1,
    )

``MultiLabelBasicNN`` operates on a single frame at a time. To work at
the example level (a frame together with its label set), we wrap the
base model in :class:`~ablkit.learning.MultiLabelABLModel`, an
``ABLModel`` subclass that threshold-binarises the sigmoid
probabilities into per-concept 0/1 pseudo-labels.

.. code:: python

    model = MultiLabelABLModel(base_model)

Building the Reasoning Part
---------------------------

The knowledge base ``BDDKB`` encodes the rules linking the 21
concepts to the 4 actions (e.g., ``red_light`` or ``obstacle`` imply
``stop``; ``green_light`` together with ``road_clear`` implies
``move_forward``). It subclasses ``KBBase``; the ``pseudo_label_list``
parameter is ``[0, 1]`` because each pseudo-label is binary, and the
``logic_forward`` method computes the 4-tuple of action labels from
the 21 concept attributes.

.. code:: python

    from reasoning.bddkb import BDDKB

    kb = BDDKB()

Since abductive reasoning is non-deterministic, multiple concept
revisions can be consistent with the ground-truth actions. The
``Reasoner`` picks the revision that minimises a user-supplied
distance function. For BDD-OIA we provide
``multi_label_confidence_dist``, which sums ``-log(p)`` over the
concept-by-concept probabilities so that revisions consistent with the
learning part's per-concept confidence are preferred:

.. code:: python

    def multi_label_confidence_dist(data_example, candidates, candidates_idxs, reasoning_results):
        pred_prob = data_example.pred_prob.T            # nc x 1
        pred_prob = np.concatenate([1 - pred_prob, pred_prob], axis=1)  # nc x 2
        cols = np.arange(len(candidates_idxs[0]))[None, :]
        corr_prob = pred_prob[cols, candidates_idxs]
        costs = -np.sum(np.log(corr_prob + 1e-6), axis=1)
        return costs

    reasoner = Reasoner(
        kb,
        dist_func=multi_label_confidence_dist,
        max_revision=3,
        require_more_revision=3,
    )

``max_revision`` and ``require_more_revision`` cap how many concept
flips the reasoner explores when searching for a consistent
revision.

Building Evaluation Metrics
---------------------------

We track two metrics. ``SymbolAccuracy`` measures how often the
predicted concepts match the ground-truth concepts, and
``BDDReasoningMetric`` measures the per-action accuracy after
passing the predicted concepts through ``logic_forward``.

.. code:: python

    metric_list = [
        SymbolAccuracy(prefix="bdd_oia"),
        BDDReasoningMetric(kb=kb, prefix="bdd_oia"),
    ]

Bridging Learning and Reasoning
-------------------------------

Finally we bridge the learning and reasoning parts via ``BDDBridge``,
a thin subclass of ``SimpleBridge`` that handles the
multi-label-specific shape of ``pred_idx`` (a ``[1, nc]`` ndarray per
example).

.. code:: python

    bridge = BDDBridge(model, reasoner, metric_list)

Training and testing reuse the standard ``SimpleBridge`` interface:

.. code:: python

    print_log("Abductive Learning on the BDD_OIA example.", logger="current")
    log_dir = ABLLogger.get_current_instance().log_dir
    weights_dir = osp.join(log_dir, "weights")

    bridge.train(
        train_data,
        loops=2,
        segment_size=0.01,
        save_interval=1,
        save_dir=weights_dir,
    )
    bridge.test(test_data)
