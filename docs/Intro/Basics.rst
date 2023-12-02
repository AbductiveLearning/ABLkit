Learn the Basics
================

In a typical Abductive Learning process, as illustrated below, 
data inputs are first mapped to pseudo labels through a machine learning model. 
These pseudo labels then pass through a knowledge base :math:`\mathcal{KB}`
to obtain the logical result by deductive reasoning. During training, 
alongside the aforementioned forward flow (i.e., prediction --> deduction reasoning), 
there also exists a reverse flow, which starts from the logical result and 
involves abductive reasoning to generate pseudo labels. 
Subsequently, these labels are processed to minimize inconsistencies with machine learning, 
which in turn revise the outcomes of the machine learning model, and then 
fed back into the machine learning model for further training. 
To implement this process, the following four steps are necessary:

.. image:: ../img/ABL-Package.png

1. Prepare datasets

    Prepare the data's input, ground truth for pseudo labels (optional), and ground truth for logical results.

2. Build machine learning part

    Build a model that defines how to map input to pseudo labels. 
    And use ``ABLModel`` to encapsulate the model.

3. Build the reasoning part

    Build a knowledge base by creating a subclass of ``KBBase``,
    and instantiate a ``ReasonerBase`` for minimizing of inconsistencies 
    between the knowledge base and pseudo labels.

4. Define Evaluation Metrics

    Define the metrics for measuring accuracy by inheriting from ``BaseMetric``.

5. Bridge machine learning and reasoning

    Use ``SimpleBridge`` to bridge the machine learning and reasoning part
    for integrated training and testing. 
