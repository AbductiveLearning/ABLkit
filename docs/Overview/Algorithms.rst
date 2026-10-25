Supported Algorithms and Usage
==============================

ABLkit provides a standard abductive-learning pipeline and several variants
for different reasoning, optimization, supervision, and label-space settings.
This page documents the public entry points and the parameters that distinguish
the variants.

The names used here follow the implementation. The earlier label
``ABL (Confidence Dict)`` corresponds to confidence-distance ABL, implemented
with ``Reasoner(dist_func="confidence")``; there is no separate
``ConfidenceDict`` class.

The 12 supported algorithms are:

1. **Hamming-distance ABL**
2. **Confidence-distance ABL**
3. **GroundKB-accelerated ABL (GroundABL)**
4. **ZOOpt-accelerated ABL**
5. **Prolog-KB ABL**
6. **Multi-label ABL**
7. **Semi-supervised ABL**
8. **Rejection-aware ABL**
9. **Similarity-based ABL (ABLSim)**
10. **A3BL (Ambiguity-aware ABL)**
11. **Verification Learning**
12. **HED Rule-learning ABL (New Rules)**

Common pipeline
---------------

Most variants use the same learning--reasoning bridge. A knowledge base and an
``ABLModel`` are connected through a ``Reasoner`` and a ``SimpleBridge``:

.. code:: python

    from ablkit.bridge import SimpleBridge
    from ablkit.reasoning import Reasoner

    reasoner = Reasoner(
        kb,
        dist_func="confidence",
        max_revision=-1,
        require_more_revision=0,
        use_zoopt=False,
    )
    bridge = SimpleBridge(model, reasoner, metric_list)
    bridge.train(train_data, loops=2, segment_size=0.01)

The main ``Reasoner`` parameters are:

* ``dist_func``: ``"hamming"``, ``"confidence"``,
  ``"avg_confidence"``, ``"similarity"``, ``"rejection"``, or a custom
  four-argument distance function.
* ``max_revision``: maximum number of revised pseudo-labels. ``-1`` means
  unrestricted; an integer sets a count; a float sets a fraction of the
  example length.
* ``require_more_revision``: number of additional revision levels searched
  after the first compatible candidates are found.
* ``use_zoopt``: enables ZOOpt-based search over revision positions.
* ``idx_to_label``: optional mapping when model indices do not follow the
  order of ``kb.pseudo_label_list``.

For a custom ``dist_func``, use the signature
``(data_example, candidates, candidate_idxs, reasoning_results)`` and return
one numerical cost per candidate.

Algorithm variants
------------------

Hamming-distance ABL
~~~~~~~~~~~~~~~~~~~~

This is the direct candidate-selection variant of the original ABL pipeline.
It selects the compatible candidate with the smallest Hamming distance from
the model prediction:

.. code:: python

    reasoner = Reasoner(
        kb,
        dist_func="hamming",
        max_revision=-1,
        require_more_revision=0,
    )

See the `MNIST Addition example <https://github.com/AbductiveLearning/ABLkit/tree/main/examples/mnist_add>`_.

Confidence-distance ABL
~~~~~~~~~~~~~~~~~~~~~~~

This variant selects candidates using the product of their per-symbol
prediction probabilities. The base model must provide ``predict_proba``:

.. code:: python

    reasoner = Reasoner(kb, dist_func="confidence")

``"confidence"`` is the default value of ``dist_func``. The related
``"avg_confidence"`` option uses the average probability instead of the
product.

GroundKB-accelerated ABL (GroundABL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GroundKB`` precomputes compatible assignments for the specified sequence
lengths and retrieves them during abduction:

.. code:: python

    from ablkit.reasoning import GroundKB, Reasoner

    kb = GroundKB(
        pseudo_label_list=list(range(10)),
        GKB_len_list=[2],
    )
    reasoner = Reasoner(kb, dist_func="confidence")

``GKB_len_list`` must contain the supported pseudo-label sequence lengths.
The MNIST Addition launcher enables this path with ``--ground``.

ZOOpt-accelerated ABL
~~~~~~~~~~~~~~~~~~~~~

Set ``use_zoopt=True`` to search over revision positions with ZOOpt:

.. code:: python

    reasoner = Reasoner(
        kb,
        dist_func="confidence",
        use_zoopt=True,
        max_revision=10,
    )

``max_revision`` bounds the number of revised positions. The default
``zoopt_budget(symbol_num)`` returns ``10 * symbol_num``; subclasses can
override it to use a different budget. The HED example uses a specialized
``HedReasoner`` with a fixed budget.

Prolog-KB ABL
~~~~~~~~~~~~~

Use ``PrologKB`` when the knowledge base is defined in a Prolog file:

.. code:: python

    from ablkit.reasoning import PrologKB, Reasoner

    kb = PrologKB(
        pseudo_label_list=list(range(10)),
        pl_file="add.pl",
    )
    reasoner = Reasoner(kb, dist_func="confidence")

The default Prolog adapter expects a rule with the form
``logic_forward(Pseudo_labels, Res)``. SWI-Prolog is also required. See the
:ref:`Prolog knowledge-base guide <prolog>` and the MNIST Addition launcher,
which enables this path with ``--prolog``.

Multi-label ABL
~~~~~~~~~~~~~~~

For binary label vectors, replace the standard learning wrappers with their
multi-label variants:

.. code:: python

    from ablkit.learning import MultiLabelABLModel, MultiLabelBasicNN

    base_model = MultiLabelBasicNN(net, loss_fn, optimizer, device="cpu")
    model = MultiLabelABLModel(base_model)

``MultiLabelABLModel`` thresholds sigmoid probabilities at ``0.5``. A custom
four-argument distance function can be supplied when the default distance
functions do not match the label representation. The
`BDD-OIA example <https://github.com/AbductiveLearning/ABLkit/tree/main/examples/bdd_oia>`_ uses this pattern with
``max_revision=3`` and ``require_more_revision=3``.

Semi-supervised ABL
~~~~~~~~~~~~~~~~~~~

Keep known pseudo-labels in ``gt_pseudo_label``, set unknown entries to
``None``, and ask ``SimpleBridge`` to use the known labels directly:

.. code:: python

    bridge.train(
        train_data=(X, pseudo_label_with_some_None, Y),
        use_supervised_data=True,
        loops=50,
        segment_size=0.01,
    )

In the MNIST Addition launcher, ``--labeled-ratio`` controls the retained
fraction and ``--seed`` controls the split. This mode is part of the standard
``SimpleBridge`` pipeline.

Rejection-aware ABL
~~~~~~~~~~~~~~~~~~~

Select the rejection-aware distance with:

.. code:: python

    reasoner = Reasoner(kb, dist_func="rejection")

The built-in ``rejection_dist`` combines confidence distance with a
normalized candidate-complexity term using ``alpha=0.5``. To change that
weight, provide a custom distance function.

Similarity-based ABL (ABLSim)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarity-based abduction uses the geometry of learned symbol embeddings:

.. code:: python

    reasoner = Reasoner(kb, dist_func="similarity")

The wrapped model must provide both ``predict_proba`` and
``extract_features(X=...)`` so that ``ABLModel`` can store embeddings for the
reasoner. The MNIST Addition model demonstrates this path with
``--dist-func similarity``.

A3BL (Ambiguity-aware ABL)
~~~~~~~~~~~~~~~~~~~~~~~~~~

A3BL aggregates multiple valid candidates into a probability-weighted soft
label instead of selecting only one candidate:

.. code:: python

    from ablkit.bridge import A3BLBridge
    from ablkit.reasoning import A3BLReasoner

    reasoner = A3BLReasoner(
        kb,
        topK=16,
        temperature=0.2,
        multi_label=False,
    )
    bridge = A3BLBridge(model, reasoner, metric_list)
    bridge.train(train_data, loops=2, segment_size=0.01)

``topK`` controls how many candidates contribute to the soft label,
``temperature`` controls their weighting, and ``multi_label=True`` selects
the binary-vector aggregation path. The MNIST Addition launcher enables A3BL
with ``--method a3bl``.

Verification Learning
~~~~~~~~~~~~~~~~~~~~~

Verification Learning enumerates high-probability assignments and trains on
each assignment that satisfies the knowledge base:

.. code:: python

    from ablkit.bridge import VerificationBridge
    from ablkit.reasoning import VerificationReasoner

    reasoner = VerificationReasoner(kb, top_k=3, max_iter=10000)
    bridge = VerificationBridge(model, reasoner, metric_list)
    bridge.train(train_data, loops=2, segment_size=0.01)

``top_k`` controls the number of valid assignments trained per segment, while
``max_iter`` caps the probability-lattice search. The ``dist_func`` parameter
is not used by ``VerificationReasoner``. The MNIST Addition launcher enables
this mode with ``--method verification --top-k 3``.

HED Rule-learning ABL (New Rules)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The HED example combines a Prolog-backed knowledge base with a specialized
reasoner and bridge that learn and validate new rules while extending equation
lengths:

.. code:: python

    from reasoning import HedKB, HedReasoner
    from bridge import HedBridge

    kb = HedKB()
    reasoner = HedReasoner(
        kb,
        dist_func="hamming",
        use_zoopt=True,
        max_revision=10,
    )
    bridge = HedBridge(model, reasoner, metric_list)
    bridge.pretrain(weights_dir)
    bridge.train(
        train_data,
        val_data,
        segment_size=10,
        min_len=5,
        max_len=8,
        save_dir=weights_dir,
    )

The HED implementation also defines the Prolog file, symbol mapping, and
staged rule-quality checks used by this example. See the complete
`HED example <https://github.com/AbductiveLearning/ABLkit/tree/main/examples/hed>`_.
