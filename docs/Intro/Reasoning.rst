`Learn the Basics <Basics.html>`_ ||
`Quick Start <Quick-Start.html>`_ ||
`Dataset & Data Structure <Datasets.html>`_ ||
`Learning Part <Learning.html>`_ ||
**Reasoning Part** ||
`Evaluation Metrics <Evaluation.html>`_ ||
`Bridge <Bridge.html>`_


Reasoning part
===============

In this section, we will look at how to build the reasoning part, which 
leverages domain knowledge and performs deductive or abductive reasoning.
In ABLkit, building the reasoning part involves two steps:

1. Build a knowledge base by creating a subclass of ``KBBase``, which
   specifies how to process pseudo-label of an example to the reasoning result.
2. Create a reasoner by instantiating the class ``Reasoner``
   to minimize inconsistencies between the knowledge base and pseudo
   labels predicted by the learning part.

.. code:: python

   from ablkit.reasoning import KBBase, GroundKB, PrologKB, Reasoner

Building a knowledge base
-------------------------

Generally, we can create a subclass derived from ``KBBase`` to build our own
knowledge base. In addition, ABLkit also offers several predefined 
subclasses of ``KBBase`` (e.g., ``PrologKB`` and ``GroundKB``), 
which we can utilize to build our knowledge base more conveniently.

Building a knowledge base from ``KBBase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the user-built KB from ``KBBase`` (a derived subclass), it's only
required to pass the ``pseudo_label_list`` parameter in the ``__init__`` method
and override the ``logic_forward`` method:

-  ``pseudo_label_list`` is the list of possible pseudo-labels (also,
   the output of the machine learning model).
-  ``logic_forward`` defines how to perform (deductive) reasoning,
   i.e. matching each example's pseudo-labels to its reasoning result. 

.. note::

   Generally, the overridden method ``logic_forward`` provided by the user accepts 
   only one parameter, ``pseudo_label`` (pseudo-labels of an example). However, for certain 
   scenarios, deductive reasoning in the knowledge base may necessitate information 
   from the input. In these scenarios, ``logic_forward`` can also accept two parameters: 
   ``pseudo_label`` and ``x``. See examples in `Zoo <../Examples/Zoo.html>`_.

After that, other operations, including how to perform abductive
reasoning, will be **automatically** set up.

MNIST Addition example
^^^^^^^^^^^^^^^^^^^^^^

As an example, the ``pseudo_label_list`` passed in MNIST Addition is all the
possible digits, namely, ``[0,1,2,...,9]``, and the ``logic_forward``
should be: “Add the two pseudo-labels to get the result.”. Therefore, the
construction of the KB (``add_kb``) for MNIST Addition would be:

.. code:: python

   class AddKB(KBBase):
      def __init__(self, pseudo_label_list=list(range(10))):
         super().__init__(pseudo_label_list)

      def logic_forward(self, pseudo_labels):
         return sum(pseudo_labels)

   add_kb = AddKB()

and (deductive) reasoning in ``add_kb`` would be:

.. code:: python

   pseudo_labels = [1, 2]
   reasoning_result = add_kb.logic_forward(pseudo_labels)
   print(f"Reasoning result of pseudo-labels {pseudo_labels} is {reasoning_result}.")

Out:
   .. code:: none
      :class: code-out

      Reasoning result of pseudo-labels [1, 2] is 3

.. _other-par:

Other optional parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

We can also pass the following parameters in the ``__init__`` method when building our
knowledge base:

-  ``max_err`` (float, optional), specifying the upper tolerance limit
   when comparing the similarity between the reasoning result of pseudo-labels 
   and the ground truth during abductive reasoning. This is only
   applicable when the reasoning result is of a numerical type. This is
   particularly relevant for regression problems where exact matches
   might not be feasible. Defaults to 1e-10. See :ref:`an example <kb-abd-2>`.
-  ``use_cache`` (bool, optional), indicating whether to use cache to store
   previous candidates (pseudo-labels generated from abductive reasoning) 
   to speed up subsequent abductive reasoning operations. Defaults to True. 
   For more information of abductive reasoning, please refer to :ref:`this <kb-abd>`.
-  ``cache_size`` (int, optional), specifying the maximum cache
   size. This is only operational when ``use_cache`` is set to True.
   Defaults to 4096.

.. _prolog:

Building a knowledge base from Prolog file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When aiming to leverage knowledge base from an external Prolog file
(which contains how to perform reasoning), we can directly create an
instance of class ``PrologKB``. Upon instantiation of
``PrologKB``, we are required to pass the ``pseudo_label_list`` (same as ``KBBase``)
and ``pl_file`` (the Prolog file) in the ``__init__`` method.

.. admonition:: What is a Prolog file?

   A Prolog file (typically have the extension ``.pl``) is a script or source 
   code file written in the Prolog language. Prolog is a logic programming language 
   where the logic is represented as facts 
   (basic assertions about some world) and 
   rules (logical statements that describe the relationships between facts). 
   A computation is initiated by running a query over these facts and rules. 
   See some Prolog examples 
   in `SWISH <https://swish.swi-prolog.org/>`_. 

After the instantiation, other operations, including how to perform
abductive reasoning, will also be **automatically** set up.

.. warning::

   Note that to use the default logic forward and abductive reasoning
   methods in this class, the Prolog (.pl) file should contain a rule
   with a strict format: ``logic_forward(Pseudo_labels, Res).``
   Otherwise, we might have to override ``logic_forward`` and
   ``get_query_string`` to allow for more adaptable usage.

MNIST Addition example (cont.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example, we can first write a Prolog file for the MNIST Addition
example as the following code, and then save it as ``add.pl``.

.. code:: prolog

   pseudo_label(N) :- between(0, 9, N).
   logic_forward([Z1, Z2], Res) :- pseudo_label(Z1), pseudo_label(Z2), Res is Z1+Z2.

Afterwards, the construction of knowledge base from Prolog file
(``add_prolog_kb``) would be as follows:

.. code:: python

   add_prolog_kb = PrologKB(pseudo_label_list=list(range(10)), pl_file="add.pl")

Building a knowledge base with GKB from ``GroundKB``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can also inherit from class ``GroundKB`` to build our own
knowledge base. In this way, the knowledge built will have a Ground KB
(GKB).

.. admonition:: What is Ground KB?

   `Ground KB <https://www.ijcai.org/proceedings/2021/250>`_ is a knowledge base prebuilt upon class initialization,
   storing all potential candidates along with their respective reasoning
   result. The key advantage of having a Ground KB is that it may
   accelerate abductive reasoning.

``GroundKB`` is a subclass of ``GKBBase``. Similar to ``KBBase``, we
are required to pass the ``pseudo_label_list`` parameter in the ``__init__`` method and
override the ``logic_forward`` method, and are allowed to pass other
:ref:`optional parameters <other-par>`. Additionally, we are required pass the
``GKB_len_list`` parameter in the ``__init__`` method.

-  ``GKB_len_list`` is the list of possible lengths for pseudo-labels of an example.

After that, other operations, including auto-construction of GKB, and
how to perform abductive reasoning, will be **automatically** set up.

MNIST Addition example (cont.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example, the ``GKB_len_list`` for MNIST Addition should be ``[2]``,
since all pseudo-labels in the example consist of two digits. Therefore,
the construction of KB with GKB (``add_ground_kb``) of MNIST Addition would be
as follows. As mentioned, the difference between this and the previously
built ``add_kb`` lies only in the base class from which it is derived
and whether an extra parameter ``GKB_len_list`` is passed.

.. code:: python

   class AddGroundKB(GroundKB):
       def __init__(self, pseudo_label_list=list(range(10)), 
                          GKB_len_list=[2]):
           super().__init__(pseudo_label_list, GKB_len_list)
           
       def logic_forward(self, nums):
           return sum(nums)
            
   add_ground_kb = AddGroundKB()

.. _kb-abd:

Performing abductive reasoning in the knowledge base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned in :ref:`What is Abductive Reasoning? <abd>`, abductive reasoning
enables the inference of candidates (i.e., possible pseudo-labels) as potential
explanations for the reasoning result. Also, in Abductive Learning where
an observation (pseudo-labels of an example predicted by the learning part) is
available, we aim to let the candidate do not largely revise the
previously identified pseudo-labels.

``KBBase`` (also, ``GroundKB`` and ``PrologKB``) implement the method
``abduce_candidates(pseudo_label, y, x, max_revision_num, require_more_revision)``
for performing abductive reasoning, where the parameters are:

-  ``pseudo_label``, pseudo-labels of an example, usually generated by the learning 
   part. They are to be revised by abductive reasoning.
-  ``y``, the ground truth of the reasoning result for the example. The
   returned candidates should be compatible with it.
- ``x``, the corresponding input example. If the information from the input 
   is not required in the reasoning process, then this parameter will not have 
   any effect.
-  ``max_revision_num``, an int value specifying the upper limit on the
   number of revised labels for each example.
-  ``require_more_revision``, an int value specifying additional number
   of revisions permitted beyond the minimum required. (e.g., If we set
   it to 0, even if ``max_revision_num`` is set to a high value, the
   method will only output candidates with the minimum possible
   revisions.)

And it returns a list of candidates (i.e., revised pseudo-labels of the example) 
that are all compatible with ``y``.

MNIST Addition example (cont.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an example, with MNIST Addition, the candidates returned by
``add_kb.abduce_candidates`` would be as follows:

+------------------+-------+----------------------+--------------------------+----------------+
| ``pseudo_label`` | ``y`` | ``max_revision_num`` | ``require_more_address`` | Output         |
+==================+=======+======================+==========================+================+
| [1,1]            | 8     | 1                    | 0                        | [[1,7], [7,1]] |
+------------------+-------+----------------------+--------------------------+----------------+
| [1,1]            | 8     | 1                    | 1                        | [[1,7], [7,1]] |
+------------------+-------+----------------------+--------------------------+----------------+
| [1,1]            | 8     | 2                    | 0                        | [[1,7], [7,1]] |
+------------------+-------+----------------------+--------------------------+----------------+
| [1,1]            | 8     | 2                    | 1                        | [[1,7],        |
|                  |       |                      |                          | [7,1], [2,6],  |
|                  |       |                      |                          | [6,2], [3,5],  |
|                  |       |                      |                          | [5,3], [4,4]]  |
+------------------+-------+----------------------+--------------------------+----------------+
| [1,1]            | 11    | 1                    | 0                        | []             |
+------------------+-------+----------------------+--------------------------+----------------+

.. _kb-abd-2:

As another example, if we set the ``max_err`` of ``AddKB`` to be 1
instead of the default 1e-10, the tolerance limit for consistency will
be higher, hence the candidates returned would be:

+------------------+-------+----------------------+--------------------------+----------------+
| ``pseudo_label`` | ``y`` | ``max_revision_num`` | ``require_more_address`` | Output         |
+==================+=======+======================+==========================+================+
| [1,1]            | 8     | 1                    | 0                        | [[1,7], [7,1], |
|                  |       |                      |                          | [1,6], [6,1],  |
|                  |       |                      |                          | [1,8], [8,1]]  |
+------------------+-------+----------------------+--------------------------+----------------+
| [1,1]            | 11    | 1                    | 0                        | [[1,9], [9,1]] |
+------------------+-------+----------------------+--------------------------+----------------+

Creating a reasoner
-------------------

After building our knowledge base, the next step is creating a
reasoner. Due to the indeterminism of abductive reasoning, there could
be multiple candidates compatible with the knowledge base. When this
happens, reasoner can minimize inconsistencies between the knowledge
base and pseudo-labels predicted by the learning part, and then return **only
one** candidate that has the highest consistency.

We can create a reasoner simply by instantiating class
``Reasoner`` and passing our knowledge base as a parameter. As an
example for MNIST Addition, the reasoner definition would be:

.. code:: python

   reasoner_add = Reasoner(kb_add)

When instantiating, besides the required knowledge base, we may also
specify:

-  ``max_revision`` (int or float, optional), specifies the upper limit
   on the number of revisions for each example when performing
   :ref:`abductive reasoning in the knowledge base <kb-abd>`. If float, denotes the
   fraction of the total length that can be revised. A value of -1
   implies no restriction on the number of revisions. Defaults to -1.
-  ``require_more_revision`` (int, optional), Specifies additional
   number of revisions permitted beyond the minimum required when
   performing :ref:`abductive reasoning in the knowledge base <kb-abd>`. Defaults to
   0.
-  ``use_zoopt`` (bool, optional), indicating whether to use the `ZOOpt library <https://github.com/polixir/ZOOpt>`_,
   which is a library for zeroth-order optimization that can be used to
   accelerate consistency minimization. Defaults to False.
-  ``dist_func`` (str, optional), specifying the distance function to be
   used when determining consistency between your prediction and
   candidate returned from knowledge base. This can be either a user-defined function
   or one that is predefined. Valid predefined options include
   “hamming”, “confidence” and “avg_confidence”. For “hamming”, it directly calculates the Hamming distance between the
   predicted pseudo-label in the data example and candidate. For “confidence”, it
   calculates the confidence distance between the predicted probabilities in the data
   example and each candidate, where the confidence distance is defined as 1 - the product
   of prediction probabilities in “confidence” and 1 - the average of prediction probabilities in “avg_confidence”.
   Defaults to “confidence”.
- ``idx_to_label`` (dict, optional), a mapping from index in the base model to label. 
   If not provided, a default order-based index to label mapping is created. 
   Defaults to None.

The main method implemented by ``Reasoner`` is
``abduce(data_example)``, which obtains the most consistent candidate 
based on the distance function defined in ``dist_func``.

MNIST Addition example (cont.)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As an example, consider these data examples for MNIST Addition:

.. code:: python

   # favor "1" for the first label
   prob1 = [[0,   0.99, 0,   0,   0,   0,   0,   0.01, 0,   0],
            [0.1, 0.1,  0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  0.1, 0.1]]

   # favor "7" for the first label
   prob2 = [[0,   0.01, 0,   0,   0,   0,   0,   0.99, 0,   0],
            [0.1, 0.1,  0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  0.1, 0.1]]

   example1 = ListData()
   example1.pred_pseudo_label = [1, 1]
   example1.pred_prob = prob1
   example1.Y = 8

   example2 = ListData()
   example2.pred_pseudo_label = [1, 1]
   example2.pred_prob = prob2
   example2.Y = 8

The compatible candidates after abductive reasoning for both examples
would be ``[[1,7], [7,1]]``. However, when the reasoner calls ``abduce`` 
to select only one candidate based on the “confidence” distance function, 
the output would differ for each example:

.. code:: python

   reasoner_add = Reasoner(kb_add, dist_func="confidence")
   candidate1 = reasoner_add.abduce(example1)
   candidate2 = reasoner_add.abduce(example2)
   print(f"The outputs for example1 and example2 are {candidate1} and {candidate2}, respectively.")

Out:
   .. code:: none
      :class: code-out

      The outputs for example1 and example2 are [1,7] and [7,1], respectively.

Specifically, as mentioned before, “confidence” calculates the distance between the data 
example and candidates based on the confidence derived from the predicted probability. 
Take ``example1`` as an example, the ``pred_prob`` in it indicates a higher 
confidence that the first label should be "1" rather than "7". Therefore, among the 
candidates [1,7] and [7,1], it would be closer to [1,7] (as its first label is "1").

