[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/AbductiveLearning/ABL-Package/blob/Dev/LICENSE)
[![flake8 Lint](https://github.com/AbductiveLearning/ABL-Package/actions/workflows/lint.yaml/badge.svg?branch=Dev)](https://github.com/AbductiveLearning/ABL-Package/actions/workflows/lint.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![ABL-Package-CI](https://github.com/AbductiveLearning/ABL-Package/actions/workflows/build-and-test.yaml/badge.svg?branch=Dev)](https://github.com/AbductiveLearning/ABL-Package/actions/workflows/build-and-test.yaml)

# ABL Package

This is the code repository of abductive learning Package.

To learn how to use it, please refer to - [document](https://www.lamda.nju.edu.cn/abl_test/docs/build/html/Overview/Abductive-Learning.html).

## Installation

ABL is distributed on `PyPI <https://pypi.org/>`__ and can be installed with ``pip``:

```bash
    # (TODO)
    $ pip install abl
```

For testing purposes, you can install it using:

```bash
    $ pip install -i https://test.pypi.org/simple/ --extra-index-url https://mirrors.nju.edu.cn/pypi/web/simple/ abl
```
    
Alternatively, to install ABL by source code, sequentially run following commands in your terminal/command line.

```bash
    $ git clone https://github.com/AbductiveLearning/ABL-Package.git
    $ cd ABL-Package
    $ pip install -v -e .
```

(Optional) If the use of a [Prolog-based knowledge base](https://www.lamda.nju.edu.cn/abl_test/docs/build/html/Intro/Reasoning.html#prolog) is necessary, the installation of [Swi-Prolog](https://www.swi-prolog.org/) is also required:

For Linux users:

```bash
    $ sudo apt-get install swi-prolog
``````

For Windows and Mac users, please refer to the [Swi-Prolog Download Page](https://www.swi-prolog.org/Download.html).

## Example 
+ MNIST ADD - [here](https://github.com/AbductiveLearning/ABL-Package/blob/Dev/examples/mnist_add)
+ Hand Written Formula - [here](https://github.com/AbductiveLearning/ABL-Package/blob/Dev/examples/hwf)
+ Hand written Equation Decipherment - [here](https://github.com/AbductiveLearning/ABL-Package/tree/Dev/examples/hed)
+ Zoo - [here](https://github.com/AbductiveLearning/ABL-Package/tree/Dev/examples/zoo)

## NOTICE 
They can only be used for academic purpose. For other purposes, please contact with LAMDA Group(www.lamda.nju.edu.cn).
