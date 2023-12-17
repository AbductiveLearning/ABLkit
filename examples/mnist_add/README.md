# MNIST Addition Example

This example shows a simple implementation of [MNIST Addition](https://link) task, where the inputs are pairs of MNIST handwritten images, and the outputs are their sums.

## Run

```
bash
pip install -r requirements.txt
python main.py
```

## Usage

```
usage: test.py [-h] [--no-cuda] [--epochs EPOCHS] [--lr LR]
               [--weight-decay WEIGHT_DECAY] [--batch-size BATCH_SIZE]
               [--loops LOOPS] [--segment_size SEGMENT_SIZE]
               [--save_interval SAVE_INTERVAL] [--max-revision MAX_REVISION]
               [--require-more-revision REQUIRE_MORE_REVISION]
               [--prolog | --ground]

MNIST Addition example

optional arguments:
  -h, --help            show this help message and exit
  --no-cuda             disables CUDA training
  --epochs EPOCHS       number of epochs in each learning loop iteration
                        (default : 1)
  --lr LR               base learning rate (default : 0.001)
  --weight-decay WEIGHT_DECAY
                        weight decay value (default : 0.03)
  --batch-size BATCH_SIZE
                        batch size (default : 32)
  --loops LOOPS         number of loop iterations (default : 5)
  --segment_size SEGMENT_SIZE
                        number of loop iterations (default : 1/3)
  --save_interval SAVE_INTERVAL
                        save interval (default : 1)
  --max-revision MAX_REVISION
                        maximum revision in reasoner (default : -1)
  --require-more-revision REQUIRE_MORE_REVISION
                        require more revision in reasoner (default : 0)
  --prolog              use PrologKB (default: False)
  --ground              use GroundKB (default: False)

```
