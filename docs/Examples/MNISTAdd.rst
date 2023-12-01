MNIST Add
==================

MNIST Add was first introduced in [1] and the inputs of this task are pairs of MNIST images and the outputs are their sums. The dataset looks like this:

.. image:: ../img/image_1.jpg
   :width: 350px
   :align: center

|

The ``gt_pseudo_label`` is only used to test the performance of the machine learning model and is not used in the training phase.

In the Abductive Learning framework, the inference process is as follows:

.. image:: ../img/image_2.jpg
   :width: 700px

[1] Robin Manhaeve, Sebastijan Dumancic, Angelika Kimmig, Thomas Demeester, and Luc De Raedt. Deepproblog: Neural probabilistic logic programming. In Advances in Neural Information Processing Systems 31 (NeurIPS), pages 3749-3759.2018.