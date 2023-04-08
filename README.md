[![flake8 Lint](https://github.com/AbductiveLearning/ABL-Package/actions/workflows/lint.yml/badge.svg?branch=Dev)](https://github.com/AbductiveLearning/ABL-Package/actions/workflows/lint.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![ABL-Package-CI](https://github.com/AbductiveLearning/ABL-Package/actions/workflows/build-and-test.yaml/badge.svg?branch=Dev)](https://github.com/AbductiveLearning/ABL-Package/actions/workflows/build-and-test.yaml)

# ABL Package

This is the code repository of abductive learning Package.

## Installation

Case a: If you develop and run abl directly, install it from source:
```bash 
git clone https://github.com/AbductiveLearning/ABL-Package.git
cd ABL-Package
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```
Case b (TO DO):  If you use abl as a dependency or third-party package, install it with pip:
```bash 
pip install abl
```

## Example 
+ MNIST ADD - [here](https://github.com/AbductiveLearning/ABL-Package/blob/Dev/examples/mnist_add/mnist_add_example.ipynb)
+ Hand Written Formula - [here](https://github.com/AbductiveLearning/ABL-Package/blob/Dev/examples/hwf/hwf_example.ipynb)
+ Hand written Equation Decipherment - [here](https://github.com/AbductiveLearning/ABL-Package/tree/Dev/examples/hed)

## Authors 

- [Yu-Xuan Huang](http://www.lamda.nju.edu.cn/huangyx/) (Nanjing University)
- [](http://www.lamda.nju.edu.cn//) (Nanjing University)


## NOTICE 
They can only be used for academic purpose. For other purposes, please contact with LAMDA Group(www.lamda.nju.edu.cn).

## To do list 

- [ ] Improve speed and accuracy
- [ ] Add comparison with DeepProbLog, NGS,... (Accuracy and Speed)
- [x] Add Inference/Abduction example with FOL engine (e.g., Prolog)
- [x] Add zoopt optimization
- [ ] Rearrange structure and make it a python package
- [ ] Documents

