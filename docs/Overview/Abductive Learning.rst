Abductive Learning
==================

Traditional supervised machine learning, e.g. classification, is
predominantly data-driven. Here, a set of training examples
:math:`\left\{\left(x_1, y_1\right), \ldots,\left(x_m, y_m\right)\right\}`
is given, where :math:`x_i \in \mathcal{X}` is the :math:`i`-th training
instance, :math:`y_i \in \mathcal{Y}` is the corresponding ground-truth
label. These data are then used to train a classifier model :math:`f:
\mathcal{X} \mapsto \mathcal{Y}` to accurately predict the unseen data.

（可能加一张图，比如左边是ML，右边是ML+KB）

In **Abductive Learning (ABL)**, we assume that, in addition to data as
examples, there is also a knowledge base :math:`\mathcal{KB}` containing
domain knowledge at our disposal. We aim for the classifier :math:`f:
\mathcal{X} \mapsto \mathcal{Y}` to make correct predictions on unseen 
data, and meanwhile, the logical facts grounded by
:math:`\left\{f(\boldsymbol{x}_1), \ldots, f(\boldsymbol{x}_m)\right\}`
should be compatible with :math:`\mathcal{KB}`.

The process of ABL is as follows:

1. Upon receiving data inputs :math:`\left\{x_1,\dots,x_m\right\}`,
   pseudo-labels
   :math:`\left\{f(\boldsymbol{x}_1), \ldots, f(\boldsymbol{x}_m)\right\}`
   are predicted by a data-driven classifier model.
2. These pseudo-labels are then converted into logical facts
   :math:`\mathcal{O}` that are acceptable for logical reasoning.
3. Conduct joint reasoning with :math:`\mathcal{KB}` to find any
   inconsistencies. If found, the logical facts that lead to minimal 
   inconsistency can be identified.
4. Modify the identified facts through abductive reasoning, returning 
   revised logical facts :math:`\Delta(\mathcal{O})` which are
   compatible with :math:`\mathcal{KB}`.
5. These revised logical facts are converted back to the form of
   pseudo-labels, and used for further learning of the classifier.
6. As a result, the classifier is updated and replaces the previous one
   in the next iteration.

This process is repeated until the classifier is no longer updated, or
the logical facts :math:`\mathcal{O}` are compatible with the knowledge
base.

The following figure illustrates this process:

一张图

We can observe that in the above figure, the left half involves machine
learning, while the right half involves logical reasoning. Thus, the
entire abductive learning process is a continuous cycle of machine
learning and logical reasoning. This effectively forms a paradigm that
is dual-driven by both data and domain knowledge, integrating and
balancing the use of machine learning and logical reasoning in a unified
model.

What is Abductive Reasoning?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


