<div align="center">

<img src="https://raw.githubusercontent.com/AbductiveLearning/ABLkit/main/docs/_static/img/logo.png" width="180">

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ablkit)](https://pypi.org/project/ablkit/) [![PyPI version](https://badgen.net/pypi/v/ablkit)](https://pypi.org/project/ablkit/) [![Documentation Status](https://readthedocs.org/projects/ablkit/badge/?version=latest)](https://ablkit.readthedocs.io/en/latest/?badge=latest) [![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/AbductiveLearning/ABLkit/blob/main/LICENSE) [![flake8 Lint](https://github.com/AbductiveLearning/ABLkit/actions/workflows/lint.yaml/badge.svg)](https://github.com/AbductiveLearning/ABLkit/actions/workflows/lint.yaml) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![ABLkit-CI](https://github.com/AbductiveLearning/ABLkit/actions/workflows/build-and-test.yaml/badge.svg)](https://github.com/AbductiveLearning/ABLkit/actions/workflows/build-and-test.yaml)

[📘Documentation](https://ablkit.readthedocs.io/en/latest/index.html) | [📚Examples](https://github.com/AbductiveLearning/ABLkit/tree/main/examples) | [💬Reporting Issues](https://github.com/AbductiveLearning/ABLkit/issues/new)

</div>

# ABLkit: A Toolkit for Abductive Learning

**ABLkit** is an efficient Python toolkit for [**Abductive Learning (ABL)**](https://www.lamda.nju.edu.cn/publication/chap_ABL.pdf). ABL is a novel paradigm that integrates machine learning and logical reasoning in a unified framework. It is suitable for tasks where both data and (logical) domain knowledge are available. 

<p align="center">
<img src="https://raw.githubusercontent.com/AbductiveLearning/ABLkit/main/docs/_static/img/ABL.png" alt="Abductive Learning" style="width: 80%;"/>
</p>

Key Features of ABLkit:

- **High Flexibility**: Compatible with various machine learning modules and logical reasoning components.
- **User-Friendly Interface**: Provide data, model, and knowledge, and get started with just a few lines of code.
- **Optimized Performance**: Optimization for high performance and accelerated training speed.

ABLkit encapsulates advanced ABL techniques, providing users with an efficient and convenient toolkit to develop dual-driven ABL systems, which leverage the power of both data and knowledge.

<p align="center">
<img src="https://raw.githubusercontent.com/AbductiveLearning/ABLkit/main/docs/_static/img/ABLkit.png" alt="ABLkit" style="width: 80%;"/>
</p>

## Installation

### Install from PyPI

The easiest way to install ABLkit is using ``pip``:

```bash
pip install ablkit
```

### Install from Source

Alternatively, to install from source code, sequentially run following commands in your terminal/command line.

```bash
git clone https://github.com/AbductiveLearning/ABLkit.git
cd ABLkit
pip install -v -e .
```

### (Optional) Install SWI-Prolog

If the use of a [Prolog-based knowledge base](https://ablkit.readthedocs.io/en/latest/Intro/Reasoning.html#prolog) is necessary, please also install [SWI-Prolog](https://www.swi-prolog.org/):

For Linux users:

```bash
sudo apt-get install swi-prolog
```

For Windows and Mac users, please refer to the [SWI-Prolog Install Guide](https://github.com/yuce/pyswip/blob/master/INSTALL.md).

## Quick Start

We use the MNIST Addition task as a quick start example. In this task, pairs of MNIST handwritten images and their sums are given, alongwith a domain knowledge base which contains information on how to perform addition operations. Our objective is to input a pair of handwritten images and accurately determine their sum.

<details>
<summary>Working with Data</summary>
<br>

ABLkit requires data in the format of `(X, gt_pseudo_label, Y)` where `X` is a list of input examples containing instances, `gt_pseudo_label` is the ground-truth label of each example in `X` and `Y` is the ground-truth reasoning result of each example in `X`. Note that `gt_pseudo_label` is only used to evaluate the machine learning model's performance but not to train it. 

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

Learning part is constructed by first defining a base model for machine learning. ABLkit offers considerable flexibility, supporting any base model that conforms to the scikit-learn style (which requires the implementation of `fit` and `predict` methods), or a PyTorch-based neural network (which has defined the architecture and implemented `forward` method). In this example, we build a simple LeNet5 network as the base model.

```python
# The 'models' module below is located in 'examples/mnist_add/'
from models.nn import LeNet5

cls = LeNet5(num_classes=10)
``` 

To facilitate uniform processing, ABLkit provides the `BasicNN` class to convert a PyTorch-based neural network into a format compatible with scikit-learn models. To construct a `BasicNN` instance, aside from the network itself, we also need to define a loss function, an optimizer, and the computing device.

```python
​import torch
​from ablkit.learning import BasicNN
​    
​loss_fn = torch.nn.CrossEntropyLoss()
​optimizer = torch.optim.RMSprop(cls.parameters(), lr=0.001, alpha=0.9)
​device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
​base_model = BasicNN(model=cls, loss_fn=loss_fn, optimizer=optimizer, device=device)
```

The base model built above is trained to make predictions on instance-level data (e.g., a single image), while ABL deals with example-level data. To bridge this gap, we wrap the `base_model` into an instance of `ABLModel`. This class serves as a unified wrapper for base models, facilitating the learning part to train, test, and predict on example-level data, (e.g., images that comprise an equation).

```python
from ablkit.learning import ABLModel
​    
​model = ABLModel(base_model)
```

</details>

<details>
<summary>Building the Reasoning Part</summary>
<br>

To build the reasoning part, we first define a knowledge base by creating a subclass of `KBBase`. In the subclass, we initialize the `pseudo_label_list` parameter and override the `logic_forward` method, which specifies how to perform (deductive) reasoning that processes pseudo-labels of an example to the corresponding reasoning result. Specifically, for the MNIST Addition task, this `logic_forward` method is tailored to execute the sum operation.

```python
from ablkit.reasoning import KBBase
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
from ablkit.reasoning import Reasoner
​    
reasoner = Reasoner(kb)
```

</details>

<details>
<summary>Building Evaluation Metrics</summary>
<br>

ABLkit provides two basic metrics, namely `SymbolAccuracy` and `ReasoningMetric`, which are used to evaluate the accuracy of the machine learning model's predictions and the accuracy of the `logic_forward` results, respectively.

```python
from ablkit.data.evaluation import ReasoningMetric, SymbolAccuracy
​    
metric_list = [SymbolAccuracy(), ReasoningMetric(kb=kb)]
```

</details>

<details>
<summary>Bridging Learning and Reasoning</summary>
<br>

Now, we use `SimpleBridge` to combine learning and reasoning in a unified ABL framework.

```python
from ablkit.bridge import SimpleBridge
​    
bridge = SimpleBridge(model, reasoner, metric_list)
```

Finally, we proceed with training and testing.

```python
​bridge.train(train_data, loops=1, segment_size=0.01)
bridge.test(test_data)
```

</details>

To explore detailed tutorials and information, please refer to - [document](https://ablkit.readthedocs.io/en/latest/index.html).

## Examples

We provide several examples in `examples/`. Each example is stored in a separate folder containing a README file.

+ [MNIST Addition](https://github.com/AbductiveLearning/ABLkit/tree/main/examples/mnist_add)
+ [Handwritten Formula (HWF)](https://github.com/AbductiveLearning/ABLkit/tree/main/examples/hwf)
+ [Handwritten Equation Decipherment](https://github.com/AbductiveLearning/ABLkit/tree/main/examples/hed)
+ [Zoo](https://github.com/AbductiveLearning/ABLkit/tree/main/examples/zoo)

## References

For more information about ABL, please refer to: [Zhou, 2019](http://scis.scichina.com/en/2019/076101.pdf) and [Zhou and Huang, 2022](https://www.lamda.nju.edu.cn/publication/chap_ABL.pdf).

```
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