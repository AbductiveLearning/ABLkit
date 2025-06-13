# BDD-OIA

This example shows an implementation of [BDD-OIA](https://twizwei.github.io/bddoia_project/) task. The BDD-OIA dataset comprises frames extracted from driving scene videos, which are utilized for autonomous driving predictions. Each frame is annotated with 4 binary labels, indicating the possible actions, namely $\textsf{move forward}$, $\textsf{stop}$, $\textsf{turn left}$, $\textsf{turn right}$. Each frame is also annotated with 21 intermediate binary concepts such as $\textsf{red light}$, $\textsf{road clear}$, etc., underlying the reasons for the possible actions.

The objective is to predict possible actions for each frame. During training, we only make use of the label supervision, along with a knowledge base, which comprises information about the relations between concepts and actions, e.g., $\textsf{red light} \lor \textsf{traffic sign} \lor \textsf{obstacle} \implies \textsf{stop}$.

Before usage, the dataset was pre-processed by [Marconato et al. (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/e560202b6e779a82478edb46c6f8f4dd-Paper-Conference.pdf) using a pretrained Faster-RCNN model on BDD-100k, in conjunction with the first module in CBM-AUC [(Sawada & Nakamura, 2022)](https://arxiv.org/abs/2202.01459), resulting in embeddings of dimension 2048.

## Run

```bash
pip install -r requirements.txt
cd dataset & unzip dataset.zip
cd ..
python main.py
```

## Usage

```bash
usage: main.py [-h] [--no-cuda] [--epochs EPOCHS] [--lr LR]
               [--batch-size BATCH_SIZE] [--loops LOOPS]
               [--segment_size SEGMENT_SIZE]
               [--save_interval SAVE_INTERVAL]
               [--max-revision MAX_REVISION]
               [--require-more-revision REQUIRE_MORE_REVISION]

BDD_OIA example

optional arguments:
  -h, --help            show this help message and exit
  --no-cuda             disables CUDA training
  --epochs EPOCHS       number of epochs in each learning loop iteration
                        (default : 1)
  --lr LR               base model learning rate (default : 0.002)
  --batch-size BATCH_SIZE
                        base model batch size (default : 32)
  --loops LOOPS
                        number of loop iterations (default : 2)
  --segment_size SEGMENT_SIZE
                        segment size (default : 0.01)
  --save_interval SAVE_INTERVAL
                        save interval (default : 1)
  --max-revision MAX_REVISION
                        maximum revision in reasoner (default : 3)
  -require-more-revision REQUIRE_MORE_REVISION
                        require more revision in reasoner (default : 3)

```
