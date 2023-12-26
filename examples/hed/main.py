import argparse
import os.path as osp

import torch
import torch.nn as nn

from abl.learning import ABLModel, BasicNN
from abl.utils import ABLLogger, print_log

from bridge import HedBridge
from consistency_metric import ConsistencyMetric
from datasets import get_dataset, split_equation
from models.nn import SymbolNet
from reasoning import HedKB, HedReasoner


def main():
    parser = argparse.ArgumentParser(description="Handwritten Equation Decipherment example")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs in each learning loop iteration (default : 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="base model learning rate (default : 0.001)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="weight decay (default : 0.0001)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="base model batch size (default : 32)"
    )
    parser.add_argument(
        "--segment_size", type=int or float, default=1000, help="segment size (default : 1000)"
    )
    parser.add_argument("--save_interval", type=int, default=1, help="save interval (default : 1)")
    parser.add_argument(
        "--max-revision",
        type=int or float,
        default=10,
        help="maximum revision in reasoner (default : 10)",
    )

    args = parser.parse_args()

    ### Working with Data
    total_train_data = get_dataset(train=True)
    train_data, val_data = split_equation(total_train_data, 3, 1)
    test_data = get_dataset(train=False)

    ### Building the Learning Part
    # Build necessary components for BasicNN
    cls = SymbolNet(num_classes=4)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(cls.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Build BasicNN
    base_model = BasicNN(
        cls,
        loss_fn,
        optimizer,
        device=device,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        stop_loss=None,
    )

    # Build ABLModel
    model = ABLModel(base_model)

    ### Building the Reasoning Part
    # Build knowledge base
    kb = HedKB()

    # Create reasoner
    reasoner = HedReasoner(kb, dist_func="hamming", use_zoopt=True, max_revision=args.max_revision)

    ### Building Evaluation Metrics
    metric_list = [ConsistencyMetric(kb=kb)]

    ### Bridge Learning and Reasoning
    bridge = HedBridge(model, reasoner, metric_list)

    # Build logger
    print_log("Abductive Learning on the HED example.", logger="current")

    # Retrieve the directory of the Log file and define the directory for saving the model weights.
    log_dir = ABLLogger.get_current_instance().log_dir
    weights_dir = osp.join(log_dir, "weights")

    bridge.pretrain(weights_dir)
    bridge.train(train_data, val_data)
    bridge.test(test_data)


if __name__ == "__main__":
    main()
