import argparse
import os.path as osp

import numpy as np
import torch
from torch import nn

from abl.bridge import SimpleBridge
from abl.data.evaluation import ReasoningMetric, SymbolAccuracy
from abl.learning import ABLModel, BasicNN
from abl.reasoning import GroundKB, KBBase, Reasoner
from abl.utils import ABLLogger, print_log

from datasets import get_dataset
from models.nn import SymbolNet


class HwfKB(KBBase):
    def __init__(
        self,
        pseudo_label_list=["1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/"],
        max_err=1e-10,
    ):
        super().__init__(pseudo_label_list, max_err)

    def _valid_candidate(self, formula):
        if len(formula) % 2 == 0:
            return False
        for i in range(len(formula)):
            if i % 2 == 0 and formula[i] not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                return False
            if i % 2 != 0 and formula[i] not in ["+", "-", "*", "/"]:
                return False
        return True

    # Implement the deduction function
    def logic_forward(self, formula):
        if not self._valid_candidate(formula):
            return np.inf
        return eval("".join(formula))


class HwfGroundKB(GroundKB):
    def __init__(
        self,
        pseudo_label_list=["1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/"],
        GKB_len_list=[1, 3, 5, 7],
        max_err=1e-10,
    ):
        super().__init__(pseudo_label_list, GKB_len_list, max_err)

    def _valid_candidate(self, formula):
        if len(formula) % 2 == 0:
            return False
        for i in range(len(formula)):
            if i % 2 == 0 and formula[i] not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                return False
            if i % 2 != 0 and formula[i] not in ["+", "-", "*", "/"]:
                return False
        return True

    # Implement the deduction function
    def logic_forward(self, formula):
        if not self._valid_candidate(formula):
            return np.inf
        return eval("".join(formula))


def main():
    parser = argparse.ArgumentParser(description="Handwritten Formula example")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="number of epochs in each learning loop iteration (default : 3)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="base model learning rate (default : 0.001)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="base model batch size (default : 128)"
    )
    parser.add_argument(
        "--loops", type=int, default=5, help="number of loop iterations (default : 5)"
    )
    parser.add_argument(
        "--segment_size", type=int, default=1000, help="segment size (default : 1000)"
    )
    parser.add_argument("--save_interval", type=int, default=1, help="save interval (default : 1)")
    parser.add_argument(
        "--max-revision",
        type=int,
        default=-1,
        help="maximum revision in reasoner (default : -1)",
    )
    parser.add_argument(
        "--require-more-revision",
        type=int,
        default=0,
        help="require more revision in reasoner (default : 0)",
    )
    parser.add_argument(
        "--ground", action="store_true", default=False, help="use GroundKB (default: False)"
    )
    parser.add_argument(
        "--max-err",
        type=float,
        default=1e-10,
        help="max tolerance during abductive reasoning (default : 1e-10)",
    )

    args = parser.parse_args()

    ### Working with Data
    train_data = get_dataset(train=True, get_pseudo_label=True)
    test_data = get_dataset(train=False, get_pseudo_label=True)

    ### Building the Learning Part
    # Build necessary components for BasicNN
    cls = SymbolNet(num_classes=13, image_size=(45, 45, 1))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cls.parameters(), lr=args.lr)
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
    )

    # Build ABLModel
    model = ABLModel(base_model)

    ### Building the Reasoning Part
    # Build knowledge base
    if args.ground:
        kb = HwfGroundKB()
    else:
        kb = HwfKB()

    # Create reasoner
    reasoner = Reasoner(
        kb, max_revision=args.max_revision, require_more_revision=args.require_more_revision
    )

    ### Building Evaluation Metrics
    metric_list = [SymbolAccuracy(prefix="hwf"), ReasoningMetric(kb=kb, prefix="hwf")]

    ### Bridge Learning and Reasoning
    bridge = SimpleBridge(model, reasoner, metric_list)

    # Build logger
    print_log("Abductive Learning on the HWF example.", logger="current")

    # Retrieve the directory of the Log file and define the directory for saving the model weights.
    log_dir = ABLLogger.get_current_instance().log_dir
    weights_dir = osp.join(log_dir, "weights")

    #  Train and Test
    bridge.train(
        train_data,
        loops=args.loops,
        segment_size=args.segment_size,
        save_interval=args.save_interval,
        save_dir=weights_dir,
    )
    bridge.test(test_data)


if __name__ == "__main__":
    main()
