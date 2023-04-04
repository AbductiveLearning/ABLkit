# `kb.py`

You can use the methods implemented in `kb.py` to construct a KB (knowledge base).

## KB Construction

When constructing a KB, users only need to specify two items:

1. `pseudo_label_list`, the output of the machine learning part or the input of the logical reasoning part.

    > For example, the `pseudo_label_list` passed in MNIST_add is [0,1,2,...,9].

2. `logic_forward`, how to get the output through the input of the logical reasoning part.
   > For example, the `logic_forward` in MNIST_add is: "Add two pseudo labels to get the result."

After that, other functions of KB (such as abduction, etc.) will be automatically constructed.

### Code Implementation

In a Python program, you can build a KB by creating a subclass of `KBBase`.

> For example, the construction of the KB (`kb1`) of MNIST_add is:
>
> ```python
> class add_KB(KBBase):
>     def __init__(self, pseudo_label_list=list(range(10))):
>         super().__init__(pseudo_label_list)
>
>     def logic_forward(self, pseudo_labels):
>         return sum(pseudo_labels)
>
> kb1 = add_KB()
> ```

## GKB

When building a KB, users can specify `GKB_flag` in `__init__` to indicate whether to build GKB (Ground Knowledge Base, domain knowledge base). GKB is a Python dictionary. The key is a list composed of `pseudo_label` input into `logic_forward`, and the value corresponding to each key is a list composed of the aforementioned `pseudo_label`. After the GKB is built, it can speed up the time required for abduction.

#### GKB Construction

When `GKB_flag` is `True`, in order to build GKB, users also need to specify `len_list`, which is used to indicate the length of each list composed of `pseudo_label` in GKB. At the same time as `__init__` constructs KB, GKB will be automatically constructed according to `pseudo_label_list`, `len_list`, and `logic_forward`.

  > For example, the `len_list` passed in MNIST_add is [2], and the GKB constructed is {0:[[0,0]], 1:[[1,0],[0,1]], 2:[[0,2],[1,1],[2,0]], ..., 18:[[9,9]]}

## Abduction

The abduction function of KB is automatically implemented in `abduce_candidates`. The following parameters need to be passed in when calling `abduce_candidates`:

- `pred_res`: the pseudo label output by machine learning
- `key`: the correct result of logical reasoning
- `max_address_num`: the maximum number of modified pseudo labels
- `require_more_address`: indicate whether to continue to increase the number of modified pseudo labels to obtain more abduction results after the result has been obtained.

The output is all possible abduction results.

> For example: The result obtained by calling `kb1.abduce_candidates` of the `kb1` (KB of MNIST_add) is as follows:
>
> |`pred_res`|`key`|`max_address_num`|`require_more_address`|Output|
> |:---:|:---:|:---:|:---:|:----|
> |[1,1]|8|2|0|[[1,7],[7,1]]|
> |[1,1]|8|2|1|[[1,7],[7,1],[2,6],[6,2],[3,5],[5,3],[4,4]]|
> |[1,1]|8|1|1|[[1,7],[7,1]]|
> |[1,1]|17|1|0|[]|
> |[1,1]|17|1|1|[[8,9],[9,8]]|
> |[1,1]|17|2|0|[[8,9],[9,8]]|
> |[1,1]|20|2|0|[]|

### Implementation of Abduction

When building KB, `GKB_flag` can be used to automatically implement abduction. If `GKB_flag` is `True`, `_abduce_by_GKB` will be called to implement abduction. Otherwise, `_abduce_by_search` will be called to implement abduction.

#### `_abduce_by_GKB`

Search whether there is a result of abduction in GKB that meets the restriction conditions composed of `pred_res`, `max_address_num`, and `require_more_address` and has a key of `key`.

> For example, in MNIST_add, when the `key` is 4, [1,3], [3,1], and [2,2] can be found in GKB. If the `pred_res` passed in at this time is [2,8], `max_address_num` is 2, and `require_more_address` is 0, the output result is [2,2].

#### `_abduce_by_search`

Starting from 0, continuously increase the number of modified pseudo labels, and enumerate all possible modified pseudo labels until the `max_address_num` is reached or the logic defined by `logic_forward` is found. Then, if `require_more_address` is not 0, continue to increase the number of modified pseudo labels and output the matching results together.

> For example, in MNIST_add, when the `key` is 4, the `pred_res` is [2,8], and the `max_address_num` is 2: when 0 pseudo labels are modified, the correct result cannot be obtained. When 1 pseudo label is modified, the possible modified logical inputs are [2,0],[2,1],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],[2,9],[0,8],[1,8],[3,8],[4,8],[5,8],[6,8],[7,8],[8,8],[9,8], among which [2,2] is the result that meets the logic. If `require_more_address` is 0, the final output result is [2,2], otherwise, continue to increase the number of modified pseudo labels and verify.

_Note: When the index of the pseudo label to be modified has been obtained using `zoopt` or other methods, it is not necessary to call the entire process of `_abduce_by_search`, only `address_by_idx` needs to be called. This part will be described in detail in `abducer_base.py`._

## `prolog_KB`

When the logic is passed in the form of a prolog program, `prolog_KB` can be directly passed to the class by specifying `pseudo_label_list` and `pl_file`.

_Note: The prolog program passed in needs to have the implementation of `logic_forward`, and the variable name of the result of `logic_forward` is `Res`._

> For example, MNIST_add can first write the `add.pl` file:
>
> ```prolog
> pseudo_label(N) :- between(0, 9, N).
> logic_forward([Z1, Z2], Res) :- pseudo_label(Z1), pseudo_label(Z2), Res is Z1+Z2.
> ```
>
> Then, the corresponding KB can be constructed
>
> ```python
> kb2 = prolog_KB(pseudo_label_list=list(range(10)), \
>                 pl_file='add.pl')
> ```

Similarly, other functions of KB (such as abduction, etc.) will be automatically constructed.

## Other Optional Parameters

The following parameters can also be passed in when building KB:

- `max_err`
- `use_cache`

### `max_err`

When the output of the logical reasoning part is a numerical value, `max_err` can be passed in so that when calling `abduce_candidates`, all results whose error with `key` is between `max_err` will be output.

> For example: The `pred_res` passed in MNIST_add is [2,2], the `key` is 7, the `max_address_num` is 2, and the `require_more_address` is 0. If `max_err` is 0, the output result is [[2,5],[5,2]]; if `max_err` is 1, the output result is [[2,4],[2,5],[2,6],[4,2],[5,2],[6,2]].

### `use_cache`

When `use_cache` is `True`, the result of each call to `abduce_candidates` will be cached so that it can be returned directly the next time it is called.

# `abducer_base.py`

You can use the methods implemented in `abducer_base.py` to help with abduction. The method is to instantiate the `AbducerBase` class and pass in `kb`.

> For example, in MNIST_add, after defining `kb1`, continue to construct
>
> ```python
> abd1 = AbducerBase(kb1)
> ```

The main function implemented by `AbducerBase` is `abduce`. Its function is to obtain **one** **most likely** result after the data is available. When calling `abduce`, the following parameters need to be passed in:

- `data`: a tuple composed of three elements, `pred_res`, `pred_res_prob`, and `key`. Among them, the definitions of `pred_res` and `key` are the same as those in `abduce_candidates` in `kb.py`, and `pred_res_prob` is the confidence list of each pseudo label output by machine learning.
- `max_address`: the maximum number of modified pseudo labels, which can be input as a float or an int. If a float is passed in, the maximum number of modified pseudo labels accounts for the proportion of all pseudo labels. If an int is passed in, it is the maximum number of modified pseudo labels (the same as the definition of `max_address_num` in `abduce_candidates` in `kb.py).
- `require_more_address`: the same definition as in `abduce_candidates` in `kb.py`.

The output is a result of abduction.

## Implementation of `abduce`

When instantiating `AbducerBase`, `zoopt` can be passed in to decide whether to use zero-order optimization to find the index of the modified pseudo label during abduction.

- When `zoopt` is `False`, zero-order optimization is not used, and `abduce_candidates` in `kb.py` is directly used to find all possible abduction results, and then `_get_one_candidate` is used to find the most likely result.
- When `zoopt` is `True`, zero-order optimization is used to find the index of the modified pseudo label, and then `address_by_idx` in `kb.py` is used to find the result of abduction. Finally, `_get_one_candidate` is used to find the most likely result.
  > For example, in MNIST_add, when `pred_res` is [2,9], `key` is 18, first use zero-order optimization to get the index of the pseudo label to be modified as 0, then substitute `pred_res`, `key`, and the modified index ([0]) into `address_by_idx` in `kb.py`. In it, all possible logical inputs after modifying the index are [0,9],[1,9],[3,9],[4,9],[5,9],[6,9],[7,9],[8,9],[9,9], among which [9,9] is the one that meets the logic, so the output is [9,9].
  >
  > Another example is HED, where `pred_res` is [1,0,1,'=',1] (`key` is set to `None` by default). First, use zero-order optimization to get the index of the pseudo label to be modified as 1, and then substitute it into `address_by_idx` in `kb.py`. In it, all possible logical inputs after modifying the index are [1,1,1,'=',1],[1,'+',1,'=',1],[1,'=',1,'=',1], among which [1,'+',1,'=',1] is the one that meets the logic, so the output is [1,'+',1,'=',1].

## What is "most likely"

When instantiating `AbducerBase`, `dist_func` can be passed in to indicate how to choose the most likely output when returning multiple results of abduction. The types that `dist_func` can choose are:

- `hamming`: Use the Hamming distance between the results after abduction and `pred_res` as the metric, and output the result with the minimum distance.
- `confidence`: Use the distance between the results after abduction and `pred_res_prob` as the metric, and output the result with the minimum distance.
  
  > For example, in MNIST_add, when `pred_res` is [1,1], `key` is 8, and `max_address` is 1, if `pred_res_prob` is [[0, 0.99, 0.01, 0, 0, 0, 0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], the output result is [1,7]; if `pred_res_prob` is [[0, 0, 0.01, 0, 0, 0, 0, 0.99, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], the output result is [7,1].

## Batch Abduction

`batch_abduce` can be used to perform abduction on a batch of data at the same time. For example, the `abd1.batch_abduce({'cls':[[1,1], [1,2]], 'prob':multiple_prob}, [4,8], max_address=2, require_more_address=0)` is called, and the output result is [[1,3], [6,2]].
