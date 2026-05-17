import argparse
import os.path as osp
import random

import torch
from torch import nn
from torch.optim import RMSprop, lr_scheduler

from ablkit.bridge import SimpleBridge
from ablkit.data.evaluation import ReasoningMetric, SymbolAccuracy
from ablkit.learning import ABLModel, BasicNN
from ablkit.reasoning import A3BLReasoner, GroundKB, KBBase, PrologKB, Reasoner
from ablkit.utils import ABLLogger, print_log

from a3bl_bridge import A3BLBridge
from datasets import get_dataset
from models.a3bl_model import A3BLBasicNN
from models.nn import LeNet5


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


DIST_FUNCS = ["hamming", "confidence", "avg_confidence", "similarity", "rejection"]


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST Addition example")
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
        "--label-smoothing",
        type=float,
        default=0.2,
        help="label smoothing in cross entropy loss (default : 0.2)",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="base model learning rate (default : 0.0003)"
    )
    parser.add_argument("--alpha", type=float, default=0.9, help="alpha in RMSprop (default : 0.9)")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="base model batch size (default : 32)"
    )
    parser.add_argument(
        "--loops", type=int, default=2, help="number of loop iterations (default : 2)"
    )
    parser.add_argument(
        "--segment_size", type=float, default=0.01, help="segment size (default : 0.01)"
    )
    parser.add_argument("--save_interval", type=int, default=1, help="save interval (default : 1)")
    parser.add_argument(
        "--max-revision", type=int, default=-1, help="maximum revision in reasoner (default : -1)"
    )
    parser.add_argument(
        "--require-more-revision",
        type=int,
        default=0,
        help="require more revision in reasoner (default : 0)",
    )
    kb_type = parser.add_mutually_exclusive_group()
    kb_type.add_argument(
        "--prolog", action="store_true", default=False, help="use PrologKB (default: False)"
    )
    kb_type.add_argument(
        "--ground", action="store_true", default=False, help="use GroundKB (default: False)"
    )
    parser.add_argument(
        "--method",
        choices=["standard", "a3bl"],
        default="standard",
        help="learning/reasoning pipeline to use (default: standard)",
    )
    parser.add_argument(
        "--dist-func",
        choices=DIST_FUNCS,
        default="confidence",
        help="distance function used by the reasoner (default: confidence)",
    )
    parser.add_argument(
        "--labeled-ratio",
        type=float,
        default=1.0,
        help=(
            "fraction in (0, 1] of training samples that keep their ground-truth pseudo-labels. "
            "Values below 1.0 enable the semi-supervised pipeline (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for semi-supervised split (default: 0)"
    )

    args = parser.parse_args()
    if not (0.0 < args.labeled_ratio <= 1.0):
        parser.error("--labeled-ratio must be in (0, 1].")
    if args.method == "a3bl" and args.labeled_ratio < 1.0:
        parser.error("--method a3bl does not support --labeled-ratio < 1.0.")
    if args.method == "a3bl" and args.dist_func == "rejection":
        parser.error("--method a3bl is not compatible with --dist-func rejection.")
    return args


def build_kb(args):
    if args.prolog:
        return PrologKB(pseudo_label_list=list(range(10)), pl_file="add.pl")
    if args.ground:
        return AddGroundKB()
    return AddKB()


def build_base_model(args):
    net = LeNet5(num_classes=10)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = RMSprop(net.parameters(), lr=args.lr, alpha=args.alpha)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        pct_start=0.15,
        epochs=args.loops,
        steps_per_epoch=int(1 / args.segment_size),
    )
    nn_cls = A3BLBasicNN if args.method == "a3bl" else BasicNN
    return nn_cls(
        net,
        loss_fn,
        optimizer,
        scheduler=scheduler,
        device=device,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )


def mask_pseudo_labels(pseudo_label, ratio, seed):
    """
    Randomly null out ``(1 - ratio)`` of the entries in ``pseudo_label`` so the
    semi-supervised pipeline treats them as unlabeled. The remaining entries
    keep their ground-truth values and are used directly during abduction.
    """
    rng = random.Random(seed)
    n = len(pseudo_label)
    keep = set(rng.sample(range(n), int(round(ratio * n))))
    return [pseudo_label[i] if i in keep else None for i in range(n)]


def main():
    args = parse_args()

    print_log("Abductive Learning on the MNIST Addition example.", logger="current")

    print_log("Working with Data.", logger="current")
    train_data = get_dataset(train=True, get_pseudo_label=True)
    test_data = get_dataset(train=False, get_pseudo_label=True)

    val_data = None
    if args.labeled_ratio < 1.0:
        X, pseudo_label, Y = train_data
        val_data = train_data
        train_data = (X, mask_pseudo_labels(pseudo_label, args.labeled_ratio, args.seed), Y)
        print_log(
            f"Semi-supervised: keeping {args.labeled_ratio:.0%} of pseudo-labels.",
            logger="current",
        )

    print_log("Building the Learning Part.", logger="current")
    base_model = build_base_model(args)
    model = ABLModel(base_model)

    print_log("Building the Reasoning Part.", logger="current")
    kb = build_kb(args)
    reasoner_cls = A3BLReasoner if args.method == "a3bl" else Reasoner
    reasoner = reasoner_cls(
        kb,
        dist_func=args.dist_func,
        max_revision=args.max_revision,
        require_more_revision=args.require_more_revision,
    )

    print_log("Building Evaluation Metrics.", logger="current")
    metric_list = [SymbolAccuracy(prefix="mnist_add"), ReasoningMetric(kb=kb, prefix="mnist_add")]

    print_log("Bridge Learning and Reasoning.", logger="current")
    bridge_cls = A3BLBridge if args.method == "a3bl" else SimpleBridge
    bridge = bridge_cls(model, reasoner, metric_list)

    log_dir = ABLLogger.get_current_instance().log_dir
    weights_dir = osp.join(log_dir, "weights")

    train_kwargs = dict(
        loops=args.loops,
        segment_size=args.segment_size,
        save_interval=args.save_interval,
        save_dir=weights_dir,
    )
    if args.method != "a3bl" and args.labeled_ratio < 1.0:
        train_kwargs["use_supervised_data"] = True
    if val_data is not None and args.method != "a3bl":
        train_kwargs["val_data"] = val_data

    bridge.train(train_data, **train_kwargs)
    bridge.test(test_data)


if __name__ == "__main__":
    main()
