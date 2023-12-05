Abductive Learning
==================

Traditional supervised machine learning, e.g. classification, is
predominantly data-driven, as shown in the below figure. 
Here, a set of data examples is given, 
where the input serving as training
instance, and the ouput serving as the corresponding ground-truth
label. These data are then used to train a classifier model :math:`f` 
to accurately predict the unseen data input.

.. image:: ../img/ML.png
   :align: center
   :width: 300px

In **Abductive Learning (ABL)**, we assume that, in addition to data as
examples, there is also a knowledge base :math:`\mathcal{KB}` containing
domain knowledge at our disposal. We aim for the classifier :math:`f` 
to make correct predictions on data input :math:`\{x_1,\dots,x_m\}`, 
and meanwhile, the logical facts grounded by
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
4. Modify the identified facts through **abductive reasoning** (or, **abduction**), 
   returning revised logical facts :math:`\Delta(\mathcal{O})` which are
   compatible with :math:`\mathcal{KB}`.
5. These revised logical facts are converted back to the form of
   pseudo-labels, and used like ground-truth labels in conventional 
   supervised learning to train a new classifier.
6. The new classifier will then be adopted to replace the previous one
   in the next iteration.

This above process repeats until the classifier is no longer updated, or
the logical facts :math:`\mathcal{O}` are compatible with the knowledge
base.

The following figure illustrates this process:

.. image:: ../img/ABL.png
   :width: 800px

We can observe that in the above figure, the left half involves machine
learning, while the right half involves logical reasoning. Thus, the
entire abductive learning process is a continuous cycle of machine
learning and logical reasoning. This effectively forms a paradigm that
is dual-driven by both data and domain knowledge, integrating and
balancing the use of machine learning and logical reasoning in a unified
model.

.. _abd:

.. admonition:: What is Abductive Reasoning?

   Abductive reasoning, also known as abduction, refers to the process of
   selectively inferring certain facts and hypotheses that explain
   phenomena and observations based on background knowledge. Unlike
   deductive reasoning, which leads to definitive conclusions, abductive
   reasoning may arrive at conclusions that are plausible but not conclusively
   proven.

   In Abductive Learning, given :math:`\mathcal{KB}` (typically expressed
   in first-order logic clauses), one can perform both deductive and 
   abductive reasoning. Deductive reasoning allows deriving
   :math:`b` from :math:`a`, while abductive reasoning allows inferring
   :math:`a` as an explanation of :math:`b`. In other words, 
   deductive reasoning and abductive reasoning differ in which end, 
   right or left, of the proposition “:math:`a\models b`” serves as conclusion.
