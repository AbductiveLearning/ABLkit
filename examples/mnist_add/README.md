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

## Environment

For all experiments, we used a single linux server. Details on the specifications are listed in the table below.

<table class="tg" style="margin-left: auto; margin-right: auto;">
<thead>
<tr>
    <th>CPU</th>
    <th>GPU</th>
    <th>Memory</th>
    <th>OS</th>
</tr>
</thead>
<tbody>
<tr>
    <td>2 * Xeon Platinum 8358, 32 Cores, 2.6 GHz Base Frequency</td>
    <td>A100 80GB</td>
    <td>512GB</td>
    <td>Ubuntu 20.04</td>
</tr>
</tbody>
</table>


## Performance

We present the results of ABL as follows, which include the reasoning accuracy (the proportion of equations that are correctly summed), and the training time used to achieve this accuracy. These results are compared with the following methods:

- [**NeurASP**](https://github.com/azreasoners/NeurASP): An extension of answer set programs by treating the neural network output as the probability distribution over atomic facts;
- [**DeepProbLog**](https://github.com/ML-KULeuven/deepproblog): An extension of ProbLog by introducing neural predicates in Probabilistic Logic Programming;
- [**LTN**](https://github.com/logictensornetworks/logictensornetworks): A neural-symbolic framework that uses differentiable first-order logic language to incorporate data and logic.
- [**DeepStochLog**](https://github.com/ML-KULeuven/deepstochlog): A neural-symbolic framework based on stochastic logic program.

<table class="tg" style="margin-left: auto; margin-right: auto;">
<thead>
<tr>
    <th>Method</th>
    <th>Accuracy</th>
    <th>Time to achieve the Acc. (s)</th>
    <th>Peak Memory Usage (MB)</th>
</tr>
</thead>
<tbody>
<tr>
    <td>NeurASP</td>
    <td>96.2</td>
    <td>966</td>
    <td>3552</td>
</tr>
<tr>
    <td>DeepProbLog</td>
    <td>97.1</td>
    <td>2045</td>
    <td>3521</td>
</tr>
<tr>
    <td>LTN</td>
    <td>97.4</td>
    <td>251</td>
    <td>3860</td>
</tr>
<tr>
    <td>DeepStochLog</td>
    <td>97.5</td>
    <td>257</td>
    <td>3545</td>
</tr>
<tr>
    <td>ABL</td>
    <td><span style="font-weight:bold">98.1</span></td>
    <td><span style="font-weight:bold">47</span></td>
    <td><span style="font-weight:bold">2482</span></td>
</tr>
</tbody>
</table>