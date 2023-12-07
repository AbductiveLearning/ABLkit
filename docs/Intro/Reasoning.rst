`Learn the Basics <Basics.html>`_ ||
`Quick Start <Quick-Start.html>`_ ||
`Dataset & Data Structure <Datasets.html>`_ ||
`Learning Part <Learning.html>`_ ||
**Reasoning Part** ||
`Evaluation Metrics <Evaluation.html>`_ ||
`Bridge <Bridge.html>`_


Reasoning part
===============

In ABL-Package, constructing the reasoning part involves two steps:

1. Build a knowledge base by creating a subclass of ``KBBase``, which
   defines how to map pseudo labels to reasoning results.
2. Define a reasoner by creating an instance of class ``Reasoner``
   to minimize inconsistencies between the knowledge base and pseudo
   labels predicted by the learning part.

Step 1: Build a knowledge base
------------------------------

Build your knowledge base from `KBBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generally, users can inherit from class ``KBBase`` to build your own
knowledge base. For the user-built KB (an inherited subclass), it's only
required to initialize the ``pseudo_label_list`` parameter
and override the ``logic_forward`` function:

-  **pseudo_label_list** is the list of possible pseudo labels (also,
   the output of the machine learning model).
-  **logic_forward** defines how to perform (deductive) reasoning,
   i.e. matching each pseudo label to their reasoning result.

After that, other operations, including how to perform abductive
reasoning, will be **automatically** set up.

MNIST Addition example
^^^^^^^^^^^^^^^^^^^^^^

As an example, the ``pseudo_label_list`` passed in MNIST Addition is all the
possible digits, namely, ``[0,1,2,...,9]``, and the ``logic_forward``
is: “Add two pseudo labels to get the result.”. Therefore, the
construction of the KB (``add_kb``) of MNIST Addition would be:

.. code:: python

   class AddKB(KBBase):
      def __init__(self, pseudo_label_list=list(range(10))):
         super().__init__(pseudo_label_list)

      def logic_forward(self, pseudo_labels):
         return sum(pseudo_labels)

   add_kb = AddKB()

.. _other-par:

Other optional parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

You can also initialize the following parameters when building your
knowledge base:

-  **max_err** (float, optional), specifying the upper tolerance limit
   when comparing the similarity between a candidate's reasoning result
   and the ground truth during abductive reasoning. This is only
   applicable when the reasoning result is of a numerical type. This is
   particularly relevant for regression problems where exact matches
   might not be feasible. Defaults to 1e-10. See :ref:`an example <kb-abd-2>`.
-  **use_cache** (bool, optional), indicating whether to use cache for
   previously abduced candidates to speed up subsequent abductive
   reasoning operations. Defaults to True. Defaults to True.
-  **cache_size** (int, optional), specifying the maximum cache
   size. This is only operational when ``use_cache`` is set to True.
   Defaults to 4096.

Diverse choices for building knowledge base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to building your own knowledge base through inheriting from class 
``KBBase``, ABL-Package also offers several predefined subclasses of ``KBBase``, 
which you can utilize to construct your knowledge base more conveniently.

Build your Knowledge base from Prolog file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For users aiming to leverage knowledge base from an external Prolog file
(which contains how to perform reasoning), they can directly create an
instance of class ``PrologKB``. Specifically, upon instantiation of
``PrologKB``, users are required to provide the ``pseudo_label_list``
and ``pl_file`` (the Prolog file).

After the instantiation, other operations, including how to perform
abductive reasoning, will also be **automatically** set up.

.. warning::

   Note that to use the default logic forward and abductive reasoning
   methods in this class, the Prolog (.pl) file should contain a rule
   with a strict format: ``logic_forward(Pseudo_labels, Res).``
   Otherwise, users might have to override ``logic_forward`` and
   ``get_query_string`` to allow for more adaptable usage.

MNIST Addition example (cont.)
"""""""""""""""""""""""""""""""

As an example, one can first write a Prolog file for the MNIST Addition
example as the following code, and then save it as ``add.pl``.

.. code:: prolog

   pseudo_label(N) :- between(0, 9, N).
   logic_forward([Z1, Z2], Res) :- pseudo_label(Z1), pseudo_label(Z2), Res is Z1+Z2.

Afterwards, the construction of knowledge base from Prolog file
(``add_prolog_kb``) would be as follows:

.. code:: python

   add_prolog_kb = PrologKB(pseudo_label_list=list(range(10)),
                            pl_file="add.pl")

Build your Knowledge base with GKB from ``GroundKB``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users can also inherit from class ``GroundKB`` to build their own
knowledge base. In this way, the knowledge built will have a Ground KB
(GKB).

.. admonition:: What is Ground KB?

   `Ground KB <https://www.ijcai.org/proceedings/2021/250>`_ is a knowledge base prebuilt upon class initialization,
   storing all potential candidates along with their respective reasoning
   result. The key advantage of having a Ground KB is that it may
   accelerate abductive reasoning.

``GroundKB`` is a subclass of ``GKBBase``. Similar to ``KBBase``, users
are required to initialize the ``pseudo_label_list`` parameter and
override the ``logic_forward`` function, and are allowed to pass other
:ref:`optional parameters <other-par>`. Additionally, users are required initialize the
``GKB_len_list`` parameter.

-  **GKB_len_list** is the list of possible lengths of pseudo label.

After that, other operations, including auto-construction of GKB, and
how to perform abductive reasoning, will be **automatically** set up.

MNIST Addition example (cont.)
"""""""""""""""""""""""""""""""

As an example, the ``GKB_len_list`` for MNIST Addition should be ``[2]``,
since all pseudo labels in the example consist of two digits. Therefore,
the construction of KB with GKB (``add_ground_kb``) of MNIST Addition would be
as follows. As mentioned, the difference between this and the previously
built ``add_kb`` lies only in the base class from which it is inherited
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

Perform abductive reasoning in your knowledge base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned in :ref:`What is Abductive Reasoning? <abd>`, abductive reasoning
enables the inference of candidate pseudo labels as potential
explanations for the reasoning result. Also, in Abductive Learning where
an observation (a pseudo label predicted by the learning part) is
available, we aim to let the candidate do not largely revise the
previously identified pseudo label.

``KBBase`` (also, ``GroundKB`` and ``PrologKB``) implement the method
``abduce_candidates(pseudo_label, y, max_revision_num, require_more_revision)``
for conducting abductive reasoning, where the parameters are:

-  **pseudo_label**, the pseudo label sample to be revised by abductive
   reasoning, usually generated by the learning part.
-  **y**, the ground truth of the reasoning result for the sample. The
   returned candidates should be compatible with it.
-  **max_revision_num**, an int value specifying the upper limit on the
   number of revised labels for each sample.
-  **require_more_revision**, an int value specifiying additional number
   of revisions permitted beyond the minimum required. (e.g., If we set
   it to 0, even if ``max_revision_num`` is set to a high value, the
   method will only output candidates with the minimum possible
   revisions.)

And it return a list of candidates (i.e., revised pseudo labels) that
are all compatible with ``y``.

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

Step 2: Create a reasoner
-------------------------

After building your knowledge base, the next step is defining a
reasoner. Due to the indeterminism of abductive reasoning, there could
be multiple candidates compatible to the knowledge base. When this
happens, reasoner can minimize inconsistencies between the knowledge
base and pseudo labels predicted by the learning part, and then return **only
one** candidate which has highest consistency.

You can create a reasoner simply by defining an instance of class
``Reasoner`` and passing your knowledge base as an parameter. As an
example for MNIST Addition, the reasoner definition would be:

.. code:: python

   reasoner_add = Reasoner(kb_add)

When instantiating, besides the required knowledge base, you may also
specify:

-  **max_revision** (int or float, optional), specifies the upper limit
   on the number of revisions for each data sample when performing
   :ref:`abductive reasoning in the knowledge base <kb-abd>`. If float, denotes the
   fraction of the total length that can be revised. A value of -1
   implies no restriction on the number of revisions. Defaults to -1.
-  **require_more_revision** (int, optional), Specifies additional
   number of revisions permitted beyond the minimum required when
   performing :ref:`abductive reasoning in the knowledge base <kb-abd>`. Defaults to
   0.
-  **use_zoopt** (bool, optional), indicating whether to use `ZOOpt library <https://github.com/polixir/ZOOpt>`_.
   It is a library for zeroth-order optimization that can be used to
   accelerate consistency minimization. Defaults to False.
-  **dist_func** (str, optional), specifying the distance function to be
   used when determining consistency between your prediction and
   candidate returned from knowledge base. Valid options include
   “confidence” (default) and “hamming”. For “confidence”, it calculates
   the distance between the prediction and candidate based on confidence
   derived from the predicted probability in the data sample. For
   “hamming”, it directly calculates the Hamming distance between the
   predicted pseudo label in the data sample and candidate.

The main method implemented by ``Reasoner`` is
``abduce(data_sample)``, which obtains the most consistent candidate.

MNIST Addition example (cont.)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As an example, consider these data samples for MNIST Addition:

.. code:: python

   # favor "1" for the first label
   prob1 = [[0,   0.99, 0,   0,   0,   0,   0,   0.01, 0,   0],
            [0.1, 0.1,  0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  0.1, 0.1]]

   # favor "7" for the first label
   prob2 = [[0,   0.01, 0,   0,   0,   0,   0,   0.99, 0,   0],
            [0.1, 0.1,  0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  0.1, 0.1]]

   sample1 = ListData()
   sample1.pred_pseudo_label = [1, 1]
   sample1.pred_prob = prob1
   sample1.Y = 8

   sample2 = ListData()
   sample2.pred_pseudo_label = [1, 1]
   sample2.pred_prob = prob2
   sample2.Y = 8

The compatible candidates after abductive reasoning for both samples
would be ``[[1,7], [7,1]]``. However, when selecting only one candidate
based on confidence, the output from ``reasoner_add.abduce`` would
differ for each sample:

=============== ======
``data_sample`` Output
=============== ======
sample1         [1,7]
sample2         [7,1]
=============== ======
