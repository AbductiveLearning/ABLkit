# Handwritten Formula

This example shows a simple implementation of [Handwritten Formula](https://arxiv.org/abs/2006.06649) task, where handwritten images of decimal formulas and their computed results are given, alongwith a domain knowledge base containing information on how to compute the decimal formula. The task is to recognize the symbols (which can be digits or operators '+', '-', '×', '÷') of handwritten images and accurately determine their results.

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Usage

```bash
usage: main.py [-h] [--no-cuda] [--epochs EPOCHS] [--lr LR]
               [--batch-size BATCH_SIZE]
               [--loops LOOPS] [--segment_size SEGMENT_SIZE]
               [--save_interval SAVE_INTERVAL] [--max-revision MAX_REVISION]
               [--require-more-revision REQUIRE_MORE_REVISION]
               [--ground] [--max-err MAX_ERR]

Handwritten Formula example

optional arguments:
  -h, --help            show this help message and exit
  --no-cuda             disables CUDA training
  --epochs EPOCHS       number of epochs in each learning loop iteration
                        (default : 1)
  --lr LR               base model learning rate (default : 0.001)
  --batch-size BATCH_SIZE
                        base model batch size (default : 32)
  --loops LOOPS         number of loop iterations (default : 5)
  --segment_size SEGMENT_SIZE
                        segment size (default : 1/3)
  --save_interval SAVE_INTERVAL
                        save interval (default : 1)
  --max-revision MAX_REVISION
                        maximum revision in reasoner (default : -1)
  --require-more-revision REQUIRE_MORE_REVISION
                        require more revision in reasoner (default : 0)
  --ground              use GroundKB (default: False)
  --max-err MAX_ERR     max tolerance during abductive reasoning (default : 1e-10)

```
