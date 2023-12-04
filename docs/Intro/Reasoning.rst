`Learn the Basics <Basics.html>`_ ||
`Quick Start <Quick-Start.html>`_ ||
`Dataset & Data Structure <Datasets.html>`_ ||
`Learning Part <Learning.html>`_ ||
**Reasoning Part** ||
`Evaluation Metrics <Evaluation.html>`_ ||
`Bridge <Bridge.html>`_


Reasoning part
===============

In ABL-Package, there are two steps to construct the reasoning part:

1. Build a knowledge base by creating a subclass of ``KBBase``, which
   defines how to map pseudo labels to logical results.
2. Define a reasoner by creating an instance of class ``ReasonerBase``
   to minimize inconsistencies between the knowledge base and pseudo
   labels.

Build a knowledge base
----------------------

Build your Knowledge base from ``KBBase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generally, users can inherit from class ``KBBase`` to build their own
knowledge base. For the user-build KB (an inherited subclass), it's only
required for the user to initialize the ``pseudo_label_list`` parameters
and override the ``logic_forward`` function:

-  ``pseudo_label_list`` is the list of possible pseudo labels (i.e.,
   the output of the machine learning model).
-  ``logic_forward`` is how to perform (deductive) reasoning,
   i.e. matching each pseudo label to their logical result.

After that, other operations, including how to perform abductive
reasoning, will be **automatically** set up.

As an example, the ``pseudo_label_list`` passed in MNISTAdd is all the
possible digits, namely, ``[0,1,2,...,9]``, and the ``logic_forward``
is: “Add two pseudo labels to get the result.”. Therefore, the
construction of the KB (``add_kb``) of MNISTAdd would be:

.. code:: python

   class AddKB(KBBase):
    def __init__(self, pseudo_label_list=list(range(10))):
        super().__init__(pseudo_label_list)

    def logic_forward(self, pseudo_labels):
        return sum(pseudo_labels)

   add_kb = AddKB()

Other optional parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

The following parameters can also be passed in when building your
knowledge base:

-  ``max_err`` (float, optional), which is the upper tolerance limit
   when comparing the similarity between a candidate's logical result
   during abductive reasoning. This is only applicable when the logical
   result is of a numerical type. This is particularly relevant for
   regression problems where exact matches might not be feasible.
   Defaults to 1e-10.
-  ``use_cache`` (bool, optional), indicates whether to use cache for
   previously abduced candidates to speed up subsequent abductive
   reasoning operations. Defaults to True.
-  ``max_cache_size`` (int, optional), The maximum cache size. This is
   only operational when ``use_cache`` is set to True. Defaults to 4096.

Build your Knowledge base with GKB from ``GroundKB``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can also inherit from class ``GroundKB`` to build their own
knowledge base. In this way, the knowledge built will have a Ground KB
(GKB).

.. admonition:: What is Ground KB?

   Ground KB is a knowledge base prebuilt upon class initialization,
   storing all potential candidates along with their respective logical
   result. The key advantage of having a Ground KB is that it may
   accelerate abductive reasoning.

Similar to ``KBBase``, users are required to initialize the
``pseudo_label_list`` parameter and override the ``logic_forward``
function. Additionally, users should initialize the ``GKB_len_list``
parameter.

-  ``GKB_len_list`` is the list of possible lengths of pseudo label.

After that, other operations, including auto-construction of GKB, and
how to perform abductive reasoning, will be **automatically** set up.

As an example, the ``GKB_len_list`` for MNISTAdd should be ``[2]``,
since all pseudo labels in the example consist of two digits. Therefore,
the construction of KB with GKB (``add_ground_kb``) of MNISTAdd would be
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

Build your Knowledge base from Prolog file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For users aiming to leverage knowledge base from an external Prolog file
(which contain how to perform reasoning), they may directly creating an
instance of class ``PrologKB``. Specifically, upon instantiation of
``PrologKB``, users are required to provide the ``pseudo_label_list``
and ``pl_file`` (the Prolog file).

After the instantiation, other operations, including how to perform
abductive reasoning, will also be **automatically** set up.

.. attention::

   Note that in order to use the default logic forward and abductive reasoning
   methods in this class ``PrologKB``, the Prolog (.pl) file should contain a rule
   with a strict format: ``logic_forward(Pseudo_labels, Res).``
   Otherwise, users might have to override ``logic_forward`` and
   ``get_query_string`` to allow for more adaptable usage.

As an example, one can first write a Prolog file for the MNISTAdd
example as the following code, and then save it as ``add.pl``.

.. code:: prolog

   pseudo_label(N) :- between(0, 9, N).
   logic_forward([Z1, Z2], Res) :- pseudo_label(Z1), pseudo_label(Z2), Res is Z1+Z2.

Afterwards, the construction of knowledge base from Prolog file
(``add_prolog_kb``) would be as follows:

.. code:: python

   add_prolog_kb = PrologKB(pseudo_label_list=list(range(10)),
                            pl_file="add.pl")

Create a reasoner
-----------------