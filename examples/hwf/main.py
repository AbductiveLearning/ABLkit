import argparse
import ast
import operator
import os.path as osp
from typing import List

import numpy as np
import torch
from torch import nn

from ablkit.bridge import SimpleBridge
from ablkit.data.evaluation import ReasoningMetric, SymbolAccuracy
from ablkit.learning import ABLModel, BasicNN
from ablkit.reasoning import GroundKB, KBBase, Reasoner
from ablkit.utils import ABLLogger, print_log

from datasets import get_dataset
from models.nn import SymbolNet

DIGITS = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
OPERATORS = {"+", "-", "*", "/"}
PSEUDO_LABEL_LIST = sorted(DIGITS) + sorted(OPERATORS)


def _is_well_formed(formula: List[str]) -> bool:
    """Return True iff ``formula`` alternates digit-operator-digit and has odd length."""
    if len(formula) % 2 == 0:
        return False
    for i, sym in enumerate(formula):
        expected = DIGITS if i % 2 == 0 else OPERATORS
        if sym not in expected:
            return False
    return True


_SAFE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _safe_eval(node: ast.AST) -> float:
    """Evaluate an arithmetic AST restricted to the four basic operators."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
        return _SAFE_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    raise ValueError(f"unsupported AST node: {type(node).__name__}")


def _evaluate(formula: List[str]) -> float:
    """Evaluate a well-formed digit/operator formula with proper precedence."""
    try:
        tree = ast.parse("".join(formula), mode="eval")
        return _safe_eval(tree)
    except (SyntaxError, ZeroDivisionError, ValueError):
        return np.inf


def _hwf_logic_forward(formula: List[str]) -> float:
    """Shared ``logic_forward`` implementation for both ``HwfKB`` variants."""
    if not _is_well_formed(formula):
        return np.inf
    return _evaluate(formula)


class HwfKB(KBBase):
    def __init__(
        self,
        pseudo_label_list: List[str] = PSEUDO_LABEL_LIST,
        max_err: float = 1e-10,
    ) -> None:
        super().__init__(pseudo_label_list, max_err)

    def logic_forward(self, formula: List[str]) -> float:
        return _hwf_logic_forward(formula)


class HwfGroundKB(GroundKB):
    def __init__(
        self,
        pseudo_label_list: List[str] = PSEUDO_LABEL_LIST,
        GKB_len_list: List[int] = [1, 3, 5, 7],
        max_err: float = 1e-10,
    ) -> None:
        super().__init__(pseudo_label_list, GKB_len_list, max_err)

    def logic_forward(self, formula: List[str]) -> float:
        return _hwf_logic_forward(formula)


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
        "--label-smoothing",
        type=float,
        default=0.2,
        help="label smoothing in cross entropy loss (default : 0.2)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="base model learning rate (default : 0.001)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="base model batch size (default : 128)"
    )
    parser.add_argument(
        "--loops", type=int, default=3, help="number of loop iterations (default : 3)"
    )
    parser.add_argument(
        "--segment_size", type=int, default=1000, help="segment size (default : 1000)"
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

    # Build logger
    print_log("Abductive Learning on the HWF example.", logger="current")

    # -- Working with Data ------------------------------
    print_log("Working with Data.", logger="current")

    train_data = get_dataset(train=True, get_pseudo_label=True)
    test_data = get_dataset(train=False, get_pseudo_label=True)

    # -- Building the Learning Part ---------------------
    print_log("Building the Learning Part.", logger="current")

    # Build necessary components for BasicNN
    net = SymbolNet(num_classes=13, image_size=(45, 45, 1))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Build BasicNN
    base_model = BasicNN(
        net,
        loss_fn,
        optimizer,
        device=device,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )

    # Build ABLModel
    model = ABLModel(base_model)

    # -- Building the Reasoning Part --------------------
    print_log("Building the Reasoning Part.", logger="current")

    # Build knowledge base
    if args.ground:
        kb = HwfGroundKB()
    else:
        kb = HwfKB()

    # Create reasoner
    reasoner = Reasoner(
        kb, max_revision=args.max_revision, require_more_revision=args.require_more_revision
    )

    # -- Building Evaluation Metrics --------------------
    print_log("Building Evaluation Metrics.", logger="current")
    metric_list = [SymbolAccuracy(prefix="hwf"), ReasoningMetric(kb=kb, prefix="hwf")]

    # -- Bridging Learning and Reasoning ----------------
    print_log("Bridge Learning and Reasoning.", logger="current")
    bridge = SimpleBridge(model, reasoner, metric_list)

    # Retrieve the directory of the Log file and define the directory for saving the model weights.
    log_dir = ABLLogger.get_current_instance().log_dir
    weights_dir = osp.join(log_dir, "weights")

    #  Train and Test
    bridge.train(
        train_data,
        val_data=test_data,
        loops=args.loops,
        segment_size=args.segment_size,
        save_interval=args.save_interval,
        save_dir=weights_dir,
    )
    bridge.test(test_data)


if __name__ == "__main__":
    main()
