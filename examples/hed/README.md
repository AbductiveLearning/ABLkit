# Handwritten Equation Decipherment

This notebook shows an implementation of [Handwritten Equation Decipherment](https://proceedings.neurips.cc/paper_files/paper/2019/file/9c19a2aa1d84e04b0bd4bc888792bd1e-Paper.pdf). In this task, the handwritten equations are given, which consist of sequential pictures of characters. The equations are generated with unknown operation rules from images of symbols ('0', '1', '+' and '='), and each equation is associated with a label indicating whether the equation is correct (i.e., positive) or not (i.e., negative). Also, we are given a knowledge base which involves the structure of the equations and a recursive definition of bit-wise operations. The task is to learn from a training set of above mentioned equations and then to predict labels of unseen equations. 

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Usage

```bash
usage: main.py [-h] [--no-cuda] [--epochs EPOCHS] [--lr LR]
               [--weight-decay WEIGHT_DECAY] [--batch-size BATCH_SIZE]
               [--loops LOOPS] [--segment_size SEGMENT_SIZE]
               [--save_interval SAVE_INTERVAL] [--max-revision MAX_REVISION]
               [--require-more-revision REQUIRE_MORE_REVISION]
               [--ground] [--max-err MAX_ERR]

Handwritten Equation Decipherment example

optional arguments:
  -h, --help            show this help message and exit
  --no-cuda             disables CUDA training
  --epochs EPOCHS       number of epochs in each learning loop iteration
                        (default : 1)
  --lr LR               base model learning rate (default : 0.001)
  --weight-decay WEIGHT_DECAY
                        weight decay (default : 0.0001)
  --batch-size BATCH_SIZE
                        base model batch size (default : 32)
  --save_interval SAVE_INTERVAL
                        save interval (default : 1)
  --max-revision MAX_REVISION
                        maximum revision in reasoner (default : 10)

```
