<div align="center">

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/abl)](https://pypi.org/project/abl/)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/AbductiveLearning/ABL-Package/blob/main/LICENSE)
[![flake8 Lint](https://github.com/AbductiveLearning/ABL-Package/actions/workflows/lint.yaml/badge.svg)](https://github.com/AbductiveLearning/ABL-Package/actions/workflows/lint.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![ABL-Package-CI](https://github.com/AbductiveLearning/ABL-Package/actions/workflows/build-and-test.yaml/badge.svg)](https://github.com/AbductiveLearning/ABL-Package/actions/workflows/build-and-test.yaml)

[📘Documentation](https://www.lamda.nju.edu.cn/abl_test/docs/build/html/index.html) |
[💬Reporting Issues](https://github.com/AbductiveLearning/ABL-Package/issues/new)

</div>

# Abductive Learning (ABL) Package

**ABL-Package** is an open source library for **Abductive Learning (ABL)**.
ABL is a novel paradigm that integrates machine learning and 
logical reasoning in a unified framework. It is suitable for tasks
where both data and (logical) domain knowledge are available. 

Key Features of ABL-Package:

- **Great Flexibility**: Adaptable to various machine learning modules and logical reasoning components.
- **User-Friendly**: Provide data, model, and KB, and get started with just a few lines of code.
- **High-Performance**: Optimization for high accuracy and fast training speed.

ABL-Package encapsulates advanced ABL techniques, providing users with
an efficient and convenient package to develop dual-driven ABL systems,
which leverage the power of both data and knowledge.

To learn how to use it, please refer to - [document](https://www.lamda.nju.edu.cn/abl_test/docs/build/html/index.html).

## Installation

### Install from PyPI

The easiest way to install ABL-Package is using ``pip``:
```bash
# (TODO)
pip install abl
```

For testing purposes, you can install it using:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://mirrors.nju.edu.cn/pypi/web/simple/ abl
```

### Install from Source

Alternatively, to install from source code, sequentially run following commands in your terminal/command line.

```bash
git clone https://github.com/AbductiveLearning/ABL-Package.git
cd ABL-Package
pip install -v -e .
```

### (Optional) Install SWI-Prolog

If the use of a [Prolog-based knowledge base](https://www.lamda.nju.edu.cn/abl_test/docs/build/html/Intro/Reasoning.html#prolog) is necessary, please also install [SWI-Prolog](https://www.swi-prolog.org/):

For Linux users:

```bash
sudo apt-get install swi-prolog
```

For Windows and Mac users, please refer to the [SWI-Prolog Install Guide](https://github.com/yuce/pyswip/blob/master/INSTALL.md).

## Examples

We provide several examples in `examples/`. Each example is stored in a separate folder containing a README file.

+ [MNIST Addition](https://github.com/AbductiveLearning/ABL-Package/blob/Dev/examples/mnist_add)
+ [Handwritten Formula](https://github.com/AbductiveLearning/ABL-Package/blob/Dev/examples/hwf)
+ [Handwritten Equation Decipherment](https://github.com/AbductiveLearning/ABL-Package/tree/Dev/examples/hed)
+ [Zoo](https://github.com/AbductiveLearning/ABL-Package/tree/Dev/examples/zoo)

## Quick Start

We use the MNIST Addition task as a quick start example. In this task, pairs of MNIST handwritten images and their sums are given, alongwith a domain knowledge base which contains information on how to perform addition operations. Our objective is to input a pair of handwritten images and accurately determine their sum.

<details>
<summary>Working with Data</summary>
<br>

ABL-Package requires data in the format of `(X, gt_pseudo_label, Y)` where `X` is a list of input examples containing instances, `gt_pseudo_label` is the ground-truth label of each example in `X` and `Y` is the ground-truth reasoning result of each example in `X`. Note that `gt_pseudo_label` is only used to evaluate the machine learning model's performance but not to train it. 

In the MNIST Addition task, the data loading looks like:

```python
# The 'datasets' module below is located in 'examples/mnist_add/'
from datasets import get_dataset
    
# train_data and test_data are tuples in the format of (X, gt_pseudo_label, Y)
train_data = get_dataset(train=True)
test_data = get_dataset(train=False)
```

</details>

<details>
<summary>Building the Learning Part</summary>
<br>

Learning part is constructed by first defining a base model for machine learning. The ABL-Package offers considerable flexibility, supporting any base model that conforms to the scikit-learn style (which requires the implementation of fit and predict methods), or a PyTorch-based neural network (which has defined the architecture and implemented forward method). In this example, we build a simple LeNet5 network as the base model.

```python
# The 'models' module below is located in 'examples/mnist_add/'
from models.nn import LeNet5

cls = LeNet5(num_classes=10)
``` 

To facilitate uniform processing, ABL-Package provides the `BasicNN` class to convert a PyTorch-based neural network into a format compatible with scikit-learn models. To construct a `BasicNN` instance, aside from the network itself, we also need to define a loss function, an optimizer, and the computing device.

```python
​import torch
​from abl.learning import BasicNN
​    
​loss_fn = torch.nn.CrossEntropyLoss()
​optimizer = torch.optim.RMSprop(cls.parameters(), lr=0.001, alpha=0.9)
​device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
​base_model = BasicNN(model=cls, loss_fn=loss_fn, optimizer=optimizer, device=device)
```

The base model built above is trained to make predictions on instance-level data (e.g., a single image), while ABL deals with example-level data. To bridge this gap, we wrap the base_model into an instance of `ABLModel`. This class serves as a unified wrapper for base models, facilitating the learning part to train, test, and predict on example-level data, (e.g., images that comprise an equation).

```python
from abl.learning import ABLModel
​    
​model = ABLModel(base_model)
```

</details>

<details>
<summary>Building the Reasoning Part</summary>
<br>

To build the reasoning part, we first define a knowledge base by creating a subclass of `KBBase`. In the subclass, we initialize the `pseudo_label_list` parameter and override the `logic_forward` method, which specifies how to perform (deductive) reasoning that processes pseudo-labels of an example to the corresponding reasoning result. Specifically, for the MNIST Addition task, this `logic_forward` method is tailored to execute the sum operation.

```python
from abl.reasoning import KBBase
​    
class AddKB(KBBase):
    def __init__(self, pseudo_label_list=list(range(10))):
        super().__init__(pseudo_label_list)

​    def logic_forward(self, nums):
        return sum(nums)
​    
kb = AddKB()
```

Next, we create a reasoner by instantiating the class `Reasoner`, passing the knowledge base as a parameter. Due to the indeterminism of abductive reasoning, there could be multiple candidate pseudo-labels compatible to the knowledge base. In such scenarios, the reasoner can minimize inconsistency and return the pseudo-label with the highest consistency.

```python
from abl.reasoning import Reasoner
​    
reasoner = Reasoner(kb)
```

</details>

<details>
<summary>Building Evaluation Metrics</summary>
<br>

ABL-Package provides two basic metrics, namely `SymbolAccuracy` and `ReasoningMetric`, which are used to evaluate the accuracy of the machine learning model's predictions and the accuracy of the `logic_forward` results, respectively.

```python
from abl.data.evaluation import ReasoningMetric, SymbolAccuracy
​    
metric_list = [SymbolAccuracy(), ReasoningMetric(kb=kb)]
```

</details>

<details>
<summary>Bridging Learning and Reasoning</summary>
<br>

Now, we use `SimpleBridge` to combine learning and reasoning in a
unified ABL framework.

```python
from abl.bridge import SimpleBridge
​    
bridge = SimpleBridge(model, reasoner, metric_list)
```

Finally, we proceed with training and testing.

```python
​bridge.train(train_data, loops=1, segment_size=0.01)
bridge.test(test_data)
```

</details>

## References

For more information about ABL, please refer to: [Zhou, 2019](http://scis.scichina.com/en/2019/076101.pdf) and [Zhou and Huang, 2022](https://www.lamda.nju.edu.cn/publication/chap_ABL.pdf).

```latex
@article{zhou2019abductive,
  title     = {Abductive learning: towards bridging machine learning and logical reasoning},
  author    = {Zhou, Zhi-Hua},
  journal   = {Science China Information Sciences},
  volume    = {62},
  number    = {7},
  pages     = {76101},
  year      = {2019}
}

@incollection{zhou2022abductive,
  title     = {Abductive Learning},
  author    = {Zhou, Zhi-Hua and Huang, Yu-Xuan},
  booktitle = {Neuro-Symbolic Artificial Intelligence: The State of the Art},
  editor    = {Pascal Hitzler and Md. Kamruzzaman Sarker},
  publisher = {{IOS} Press},
  pages     = {353--369},
  address   = {Amsterdam},
  year      = {2022}
}
```