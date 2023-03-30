# `kb.py`

可以使用`kb.py`中实现的方法构建 KB (知识库).

## KB 的建立

用户在建立一个 KB 时只需要指定两项：

1. `pseudo_label_list`, 即伪标记 (机器学习部分的输出/逻辑推理部分的输入) 有哪些

    > 例如, MNIST_add 传入的`pseudo_label_list`为 [0,1,2,...,9].

2. `logic_forward`, 即如何通过逻辑推理部分的输入得到输出
   > 例如, MNIST_add 中`logic_forward`为: "将两个 pseudo labels 相加得到结果"

之后, KB 的其他功能 (如反绎等) 会自动构建.

### 代码实现

在 Python 程序中, 可以通过建立一个 `KBBase` 的子类来建立 KB.

> 例如, MNIST_add 的 KB (`kb1`) 的建立为:
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

建立 KB 时, 用户可以在`__init__`中指定`GKB_flag`, 说明是否需要建立GKB (Ground Knowledge Base, 领域知识库). GKB 是一个 Python 字典, key 为`pseudo_label`组成的 list 代入`logic_forward`得到的所有可能的结果, 每个 key 对应的 value 为前述的`pseudo_label`组成的 list. 建立好 GKB 之后可以加快反绎的速度.

### GKB 的建立

`GKB_flag`为`True`时, 为了建立 GKB, 用户还需要指定`len_list`, 用于说明 GKB 的每一个`pseudo_label`组成的 list 的长度. 在``__init__``建立 KB 的同时会根据 `pseudo_label_list`、`len_list`和`logic_forward`来自动建立 GKB.
  
  > 例如, MNIST_add 传入的`len_list`为 [2], 则建立的 GKB 为 {0:[[0,0]], 1:[[1,0],[0,1]], 2:[[0,2],[1,1],[2,0]], ..., 18:[[9,9]]}

## 反绎

KB 的反绎功能在`abduce_candidates`中自动实现. 在调用`abduce_candidates`中需要传入以下参数:

- `pred_res`: 机器学习的输出的伪标记
- `key`: 逻辑推理的正确结果
- `max_address_num`: 最多修改的伪标记的个数
- `require_more_address`: 在已经反绎出结果之后, 指明是否需要继续增加修改伪标记的个数来继续得到更多的反绎结果.

得到的输出为所有可能的反绎的结果.

> 例如: 上文定义的`kb1` (MNIST_add 的 KB) 调用`kb1.abduce_candidates`得到的结果如下表:
>
> |`pred_res`|`key`|`max_address_num`|`require_more_address`|输出|
> |:---:|:---:|:---:|:---:|:----|
> |[1,1]|8|2|1|[[1,7],[7,1],[2,6],[6,2],[3,5],[5,3],[4,4]]|
> |[1,1]|8|2|0|[[1,7],[7,1]]|
> |[1,1]|8|1|1|[[1,7],[7,1]]|
> |[1,1]|17|1|1|[]|
> |[1,1]|17|2|0|[[8,9],[9,8]]|
> |[1,1]|20|2|0|[]|

### 反绎的实现

当`GKB_flag`为`True`时会调用`_abduce_by_GKB`进行反绎, 否则会调用`_abduce_by_search`进行反绎.

#### `_abduce_by_GKB`

搜索 GKB 中是否存在 key 为`key`, 且满足`pred_res`, `max_address_num`和`require_more_address`组成的限制条件的反绎结果.

> 比如, MNIST_add 中, 传入的`key`为 4, 此时在 GKB 中可以找到 [1,3], [3,1] 和 [2,2]. 如果此时传入的`pred_res`为 [2,8], `max_address_num`为 2, `require_more_address`为 0, 则输出的结果为 [2,2].

#### `_abduce_by_search`

从 0 开始不断增加修改伪标记的个数, 通过枚举得到所有可能的修改后的伪标记, 直到达到`max_address_num`或找到符合`logic_forward`定义的逻辑的结果. 接着, 如果`require_more_address`不为0就继续增加修改伪标记的个数, 将符合的结果一起输出.

> 比如, MNIST_add 中, 传入的`key`为 4, `pred_res`为 [2,8], `max_address_num`为 2: 当修改 0 个伪标记时, 不能得到正确的结果. 当修改 1 个伪标记时, 可能的修改后逻辑输入为 [2,0],[2,1],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],[2,9],  [0,8],[1,8],[3,8],[4,8],[5,8],[6,8],[7,8],[8,8],[9,8], 其中, [2,2] 符合逻辑的结果, 如果传入的`require_more_address`是 0, 则最终输出的结果就是 [2,2], 否则, 继续增加修改的伪标记的个数并检验.

_注: 如果使用 prolog 作为 KB, `_abduce_by_search`不会手动地枚举所有的可能修改后的伪标记, prolog 程序运行时会有一些剪枝等操作可以加速反绎. 关于这一部分, 将在下一小节`prolog_KB`中详述._

_注: 当使用`zoopt`或其他方式已经获得了需要修改的 index 时, 不需要调用整个`_abduce_by_search`的流程, 只需要调用其中的`address_by_idx`即可. 关于这一部分, 将在`abducer_base.py`中详述._

## `prolog_KB`

`prolog_KB` 是 `KBBase` 的子类, 当逻辑是以 prolog 程序的形式传入时, 可以直接向 `prolog_KB` 类传入`pseudo_label_list`和`pl_file` 即可构建 KB.

_需要注意: 传入的 prolog 程序中需要有 `logic_forward` 的实现, 且 `logic_forward` 结果的变量名为 `Res`._

> 比如, MNIST_add 可以首先写好`add.pl`文件:
>
> ```prolog
> pseudo_label(N) :- between(0, 9, N).
> logic_forward([Z1, Z2], Res) :- pseudo_label(Z1), pseudo_label(Z2), Res is Z1+Z2.
> ```
>
> 然后, 建立相应 KB 即可
>
> ```python
> kb2 = prolog_KB(pseudo_label_list=list(range(10)), \
>                 pl_file='add.pl')
> ```

同样地, KB 的其他功能 (如反绎等) 会自动构建.

## 其他可选参数

建立 KB 时还可以传入下列参数:

- `max_err`
- `use_cache`

### `max_err`

当逻辑推理部分的输出为数值时, 可以传入`max_err`, 使得调用`abduce_candidates`时, 只要满足与`key`的误差在`max_err`之间的结果都会被输出.

> 例如: 上文定义的`kb1` (MNIST_add 的 KB), 当`pred_res`为 [2,2], `key`为 7, `max_address_num`为 2, `require_more_address`为 0 时, 如果`max_err`为 0, 则输出的结果为 [[2,5],[5,2]]; 如果`max_err`为 1, 则输出的结果为 [[2,4],[2,5],[2,6],[4,2],[5,2],[6,2]].

### `use_cache`

当`use_cache`为`True`时, 会将每次调用`abduce_candidates`的结果缓存起来, 以便下次调用时直接返回.

## 学习逻辑规则

在`HED_prolog_KB`里面有例子.

# `abducer_base.py`

可以使用`abducer_base.py`中实现的方法帮助进行反绎. 使用的方法为实例化 `AbducerBase` 类, 并向其中传入 `kb` 即可. 

> 比如, MNIST_add 中可以在定义 `kb1` 之后, 继续构建
>
> ```python
> abd1 = AbducerBase(kb1)
> ```

`AbducerBase` 主要实现的函数是 `abduce`. 它的功能是有了数据之后得到**一个** **最有可能的**反绎的结果. 在调用`abduce`中需要传入以下参数:

- `data`: 三元素组成的 tuple, 三个元素分别为 `pred_res`, `pred_res_prob`, `key`. 其中, `pred_res`和`key`的定义同`kb.py`中的`abduce_candidates`, 而 `pred_res_prob` 是机器学习输出的每个伪标记的置信度列表.
- `max_address`: 最多修改的伪标记的数量, 可以以 float 或 int 的形式输入. 如果传入 float 最多修改的伪标记占所有伪标记的比重, 如果传入 int 为最多修改伪标记的个数 (此时同`kb.py`中`abduce_candidates`中`max_address_num`的定义)
- `require_more_address`: 定义同`kb.py`中的 `abduce_candidates`同一参数.

输出为一个反绎的结果.

## `abduce` 的实现

在实例化`AbducerBase`时可以传入`zoopt`参数, 决定在反绎过程中是否使用零阶优化找到修改的伪标记的 index.

- 当`zoopt`为`False`时, 不使用零阶优化, 直接使用`kb.py`中的`abduce_candidates`找到所有可能的反绎结果, 然后使用`_get_one_candidate`找到最有可能的反绎结果.

- 当`zoopt`为`True`时, 使用零阶优化找到修改的伪标记的 index, 然后使用`kb.py`中的`address_by_idx`找到反绎结果, 最后使用`_get_one_candidate`找到最有可能的反绎结果.
  > 比如, MNIST_add 中的`pred_res`为 [2,9], `key`为 18, 首先使用零阶优化会得到应该修改的伪标记的 index 为 0, 接着代入`pred_res`, `key` 和修改的 index([0]) 到`kb.py`的`address_by_idx`中, 在其中会先得到修改 index 处的伪标记后所有可能的逻辑输入为 [0,9],[1,9],[3,9],[4,9],[5,9],[6,9],[7,9],[8,9],[9,9], 其中 [9,9] 是符合逻辑的, 则输出 [9,9].
  >
  > 再比如, HED 中`pred_res`为[1,0,1,'=',1], (`key`默认设置为`None`). 首先使用零阶优化会得到应该修改的伪标记的 index 为 1, 代入到`kb.py`的`address_by_idx`可以得到修改 index 处的伪标记后所有可能的逻辑输入为 [1,1,1,'=',1],[1,'+',1,'=',1],[1,'=',1,'=',1], 其中 [1,'+',1,'=',1] 是符合逻辑的, 则输出 [1,'+',1,'=',1].

## 何为“最有可能的”

在实例化`AbducerBase`时需要传入`dist_func`参数, 从而使得当返回反绎结果时, 如何选择最有可能的一个输出. `dist_func` 可以选择的类型有:

- `hamming`: 用反绎后的结果与`pred_res`之间的汉明距离作为度量, 输出距离最小的反绎结果.
- `confidence`: 用反绎后的结果与`pred_res_prob`之间的距离作为度量, 输出距离最小的反绎结果.
  
  > 比如, MNIST_add 中的`pred_res`为 [1,1], `key` 为 8, `max_address`为 1, 如果`pred_res_prob`为 [[0, 0.99, 0.01, 0, 0, 0, 0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], 则输出的结果为 [1,7], 如果`pred_res_prob`为 [[0, 0, 0.01, 0, 0, 0, 0, 0.99, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], 则输出的结果为 [7,1].

## 批量化反绎

可以使用`batch_abduce`同时传入一批数据进行反绎, 如上文定义的 `abd1`, 调用`abd.batch_abduce({'cls':[[1,1], [1,2]], 'prob':multiple_prob}, [4,8], max_address=2, require_more_address=0)`时, 返回的结果为 [[1,3], [6,2]].
