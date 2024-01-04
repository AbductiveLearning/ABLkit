# Zoo Example

This example shows a simple implementation of [Zoo](https://archive.ics.uci.edu/dataset/111/zoo). In this task, attributes of animals (such as presence of hair, eggs, etc.) and their targets (the animal class they belong to) are given, along with a knowledge base which contain information about the relations between attributes and targets, e.g., Implies(milk == 1, mammal == 1). The goal of this task is to develop a learning model that can predict the targets of animals based on their attributes.

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Usage

```bash
usage: main.py [-h] [--loops LOOPS] 

Zoo example

optional arguments:
  -h, --help            show this help message and exit
  --loops LOOPS         number of loop iterations (default : 3)

```
