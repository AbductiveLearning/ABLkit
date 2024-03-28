# Handwritten Formula

This example shows a simple implementation of [Handwritten Formula](https://arxiv.org/abs/2006.06649) task, where handwritten images of decimal formulas and their computed results are given, alongwith a domain knowledge base containing information on how to compute the decimal formula. The task is to recognize the symbols (which can be digits or operators '+', '-', '×', '÷') of handwritten images and accurately determine their results.

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Usage

```bash
usage: main.py [-h] [--no-cuda] [--epochs EPOCHS]
               [--label_smoothing LABEL_SMOOTHING] [--lr LR] 
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
  --label_smoothing LABEL_SMOOTHING
                        label smoothing in cross entropy loss (default : 0.2)
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

We present the results of ABL as follows, which include the reasoning accuracy (for different equation lengths in the HWF dataset), training time (to achieve the accuracy using all equation lengths), and average memory usage (using all equation lengths). These results are compared with the following methods:

- [**NGS**](https://github.com/liqing-ustc/NGS): A neural-symbolic framework that uses a grammar model and a back-search algorithm to improve its computing process;
- [**DeepProbLog**](https://github.com/ML-KULeuven/deepproblog/tree/master): An extension of ProbLog by introducing neural predicates in Probabilistic Logic Programming;
- [**DeepStochLog**](https://github.com/ML-KULeuven/deepstochlog/tree/main): A neural-symbolic framework based on stochastic logic program.

<table class="tg" style="margin-left: auto; margin-right: auto;">
<thead>
  <tr>
    <th rowspan="2"></th>
    <th colspan="5">Reasoning Accuracy<br><span style="font-weight: normal; font-size: smaller;">(for different equation lengths)</span></th>
    <th rowspan="2">Training Time (s)<br><span style="font-weight: normal; font-size: smaller;">(to achieve the Acc. using all lengths)</span></th>
    <th rowspan="2">Average Memory Usage (MB)<br><span style="font-weight: normal; font-size: smaller;">(using all lengths)</span></th>
  </tr>
  <tr>
    <th>1</th>
    <th>3</th>
    <th>5</th>
    <th>7</th>
    <th>All</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>NGS</td>
    <td>91.2</td>
    <td>89.1</td>
    <td>92.7</td>
    <td>5.2</td>
    <td>98.4</td>
    <td>426.2</td>
    <td>3705</td>
  </tr>
  <tr>
    <td>DeepProbLog</td>
    <td>90.8</td>
    <td>85.6</td>
    <td>timeout*</td>
    <td>timeout</td>
    <td>timeout</td>
    <td>timeout</td>
    <td>4315</td>
  </tr>
  <tr>
    <td>DeepStochLog</td>
    <td>92.8</td>
    <td>87.5</td>
    <td>92.1</td>
    <td>timeout</td>
    <td>timeout</td>
    <td>timeout</td>
    <td>4355</td>
  </tr>
  <tr>
    <td>ABL</td>
    <td><span style="font-weight:bold">94.0</span></td>
    <td><span style="font-weight:bold">89.7</span></td>
    <td><span style="font-weight:bold">96.5</span></td>
    <td><span style="font-weight:bold">97.2</span></td>
    <td><span style="font-weight:bold">99.2</span></td>
    <td><span style="font-weight:bold">77.3</span></td>
    <td><span style="font-weight:bold">3074</span></td>
  </tr>
</tbody>
</table>
<p style="font-size: 13px;">* timeout: need more than 1 hour to execute</p>
