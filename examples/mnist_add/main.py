import os
import os.path as osp
import argparse

import torch
from torch import nn

from abl.bridge import SimpleBridge
from abl.evaluation import ReasoningMetric, SymbolMetric
from abl.learning import ABLModel, BasicNN
from abl.reasoning import KBBase, GroundKB, PrologKB, Reasoner
from abl.utils import ABLLogger, print_log
from examples.mnist_add.datasets import get_mnist_add
from examples.models.nn import LeNet5

class AddKB(KBBase):
    def __init__(self, pseudo_label_list=list(range(10))):
        super().__init__(pseudo_label_list)

    def logic_forward(self, nums):
        return sum(nums)

class AddGroundKB(GroundKB):
    def __init__(self, pseudo_label_list=list(range(10)), GKB_len_list=[2]):
        super().__init__(pseudo_label_list, GKB_len_list)

    def logic_forward(self, nums):
        return sum(nums)

def main():
    parser = argparse.ArgumentParser(description='MNIST Addition example')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs in each learning loop iteration (default : 1)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='base learning rate (default : 0.001)')
    parser.add_argument('--weight-decay', type=int, default=3e-2,
                        help='weight decay value (default : 0.03)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size (default : 32)')
    parser.add_argument('--loops', type=int, default=5,
                        help='number of loop iterations (default : 5)')
    parser.add_argument('--segment_size', type=int or float, default=1/3,
                        help='number of loop iterations (default : 1/3)')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='save interval (default : 1)')
    parser.add_argument('--max-revision', type=int or float, default=-1,
                        help='maximum revision in reasoner (default : -1)')
    parser.add_argument('--require-more-revision', type=int, default=5,
                        help='require more revision in reasoner (default : 0)')
    kb_type = parser.add_mutually_exclusive_group()
    kb_type.add_argument("--prolog", action="store_true", default=False,
                        help='use PrologKB (default: False)')
    kb_type.add_argument("--ground", action="store_true", default=False,
                        help='use GroundKB (default: False)')

    args = parser.parse_args()
    
    # Build logger
    print_log("Abductive Learning on the MNIST Addition example.", logger="current")

    # Retrieve the directory of the Log file and define the directory for saving the model weights.
    log_dir = ABLLogger.get_current_instance().log_dir
    weights_dir = osp.join(log_dir, "weights")

    ### Learning Part
    # Build necessary components for BasicNN
    cls = LeNet5(num_classes=10)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cls.parameters(), lr=args.lr)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Build BasicNN
    # The function of BasicNN is to wrap NN models into the form of an sklearn estimator
    base_model = BasicNN(
        cls,
        loss_fn,
        optimizer,
        device=device,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )

    # Build ABLModel
    # The main function of the ABL model is to serialize data and
    # provide a unified interface for different machine learning models
    model = ABLModel(base_model)
    
    if args.prolog:
        kb = PrologKB(pseudo_label_list=list(range(10)), pl_file="add.pl")
    elif args.ground:
        kb = AddGroundKB()
    else:
        kb = AddKB()
    reasoner = Reasoner(kb, dist_func="confidence", max_revision=args.max_revision, require_more_revision=args.require_more_revision)

    ### Datasets and Evaluation Metrics
    # Get training and testing data
    train_data = get_mnist_add(train=True, get_pseudo_label=True)
    test_data = get_mnist_add(train=False, get_pseudo_label=True)

    # Set up metrics
    metric_list = [SymbolMetric(prefix="mnist_add"), ReasoningMetric(kb=kb, prefix="mnist_add")]

    ### Bridge Machine Learning and Logic Reasoning
    bridge = SimpleBridge(model, reasoner, metric_list)

    #  Train and Test
    bridge.train(train_data, loops=args.loops, segment_size=args.segment_size, save_interval=args.save_interval, save_dir=weights_dir)
    bridge.test(test_data)

    


if __name__ == "__main__":
    main()
