**Learn the Basics** ||
`Quick Start <Quick-Start.html>`_ ||
`Dataset & Data Structure <Datasets.html>`_ ||
`Learning Part <Learning.html>`_ ||
`Reasoning Part <Reasoning.html>`_ ||
`Evaluation Metrics <Evaluation.html>`_ ||
`Bridge <Bridge.html>`_ 

Learn the Basics
================

Modules in ABL-Package
----------------------

ABL-Package is an implementation of `Abductive Learning <../Overview/Abductive-Learning.html>`_, 
designed to harmoniously integrate and balance the use of machine learning and
logical reasoning within a unified model. As depicted below, the
ABL-Package comprises three primary modules: **Data**, **Learning**, and
**Reasoning**, corresponding to the three pivotal components in current
AI: data, models, and knowledge.

.. image:: ../img/ABL-Package.png

**Data** module manages the storage, operation, and evaluation of data.
It first features class ``ListData`` (inherited from base class
``BaseDataElement``), which defines the data structures used in
Abductive Learning, and comprises common data operations like insertion,
deletion, retrieval, slicing, etc. Additionally, a series of Evaluation
Metrics, including class ``SymbolMetric`` and ``ReasoningMetric`` (both
specialized metrics derived from base class ``BaseMetric``), outline
methods for evaluating model quality from a data perspective.

**Learning** module is responsible for the construction, deployment, and
training of machine learning models. In this module, the class
``ABLModel`` is the central class that encapsulates the machine learning
model, which may incorporate models such as those based on Scikit-learn
or a neural network framework using constructed by class ``BasicNN``.

**Reasoning** module consists of the reasoning part of the Abductive
learning. The class ``KBBase`` allows users to define domain
knowledge base. For diverse types of knowledge, we also offer
implementations like ``GroundKB`` and ``PrologKB``, e.g., the latter
enables knowledge base to be imported in the form of a Prolog files.
Upon building the knowledge base, the class ``Reasoner`` is
responsible for minimizing the inconsistency between the knowledge base
and learning models.

Finally, the integration of these three modules occurs through
**Bridge** module, which features class ``SimpleBridge`` (inherited from base
class ``BaseBridge``). Bridge module synthesize data, learning, and
reasoning, and facilitate the training and testing of the entire
Abductive Learning framework.

Use ABL-Package Step by Step
----------------------------

In a typical Abductive Learning process, as illustrated below, 
data inputs are first predicted by a machine learning model, and the outcomes are a pseudo label 
sample (which consists of multiple pseudo labels). 
These labels then pass through a knowledge base :math:`\mathcal{KB}`
to obtain the reasoning result by deductive reasoning. During training, 
alongside the aforementioned forward flow (i.e., prediction --> deduction reasoning), 
there also exists a reverse flow, which starts from the reasoning result and 
involves abductive reasoning to generate possible pseudo label samples. 
Subsequently, these samples are processed to minimize inconsistencies with machine learning, 
which in turn revise the outcomes of the machine learning model, and then 
fed back into the machine learning model for further training. 
To implement this process, the following five steps are necessary:

.. image:: ../img/usage.png

1. Prepare datasets

    Prepare the data's input, ground truth for pseudo labels (optional), and ground truth for reasoning results.

2. Build the learning part

    Build a machine learning model that can predict inputs to pseudo labels. 
    Then, use ``ABLModel`` to encapsulate the model.

3. Build the reasoning part

    Define a knowledge base by building a subclass of ``KBBase``, specifying how to 
    map pseudo label samples to reasoning results.
    Also, create a ``Reasoner`` for minimizing of inconsistencies 
    between the knowledge base and the learning part.

4. Define Evaluation Metrics

    Define the metrics for measuring accuracy by inheriting from ``BaseMetric``.

5. Bridge learning and reasoning

    Use ``SimpleBridge`` to bridge the learning and reasoning part
    for integrated training and testing. 
