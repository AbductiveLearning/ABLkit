# MNIST Addition

This example shows a simple implementation of [MNIST Addition](https://arxiv.org/abs/1805.10872) task, where pairs of MNIST handwritten images and their sums are given, alongwith a domain knowledge base containing information on how to perform addition operations. The task is to recognize the digits of handwritten images and accurately determine their sum.

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Usage

```bash
usage: main.py [-h] [--no-cuda] [--epochs EPOCHS] 
               [--label_smoothing LABEL_SMOOTHING] [--lr LR] 
               [--alpha ALPHA] [--batch-size BATCH_SIZE]
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
  --label_smoothing LABEL_SMOOTHING
                        label smoothing in cross entropy loss (default : 0.2)
  --lr LR               base model learning rate (default : 0.001)
  --alpha ALPHA         alpha in RMSprop (default : 0.9)
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
  --prolog              use PrologKB (default: False)
  --ground              use GroundKB (default: False)

```


## Performance

We present the results of ABL as follows, which include the reasoning accuracy (the proportion of equations that are correctly summed), and the training time used to achieve this accuracy. These results are compared with the following methods:

- [**NeurASP**](https://github.com/azreasoners/NeurASP): An extension of answer set programs by treating the neural network output as the probability distribution over atomic facts;
- [**DeepProbLog**](https://github.com/ML-KULeuven/deepproblog): An extension of ProbLog by introducing neural predicates in Probabilistic Logic Programming;
- [**DeepStochLog**](https://github.com/ML-KULeuven/deepstochlog): A neural-symbolic framework based on stochastic logic program.

|  Method      | Accuracy | Time to achieve the Acc. (s) |
| :----------: | :------: | :--------------------------: |
|  NeurASP     |  0.964   |             354              |
| DeepProbLog  |  0.965   |             1965             |
| DeepStochLog |  0.975   |             727              |
|     ABL      |  0.980   |              42              |