import argparse
import os.path as osp
import numpy as np
import torch
from torch import optim
import torch.nn as nn

from ablkit.data.evaluation import SymbolAccuracy
from ablkit.reasoning import Reasoner
from ablkit.utils import ABLLogger, print_log

from models.nn import ConceptNet
from models.bdd_nn import BDDNN
from models.bdd_model import BDDABLModel
from reasoning.bddkb import BDDKB
from dataset.data_util import get_dataset
from bridge import BDDBridge
from metric import BDDReasoningMetric


def multi_label_confidence_dist(data_example, candidates, candidates_idxs, reasoning_results):
    pred_prob = data_example.pred_prob.T  # nc x 1
    pred_prob = np.concatenate([1 - pred_prob, pred_prob], axis=1)  # nc x 2
    cols = np.arange(len(candidates_idxs[0]))[None, :]
    corr_prob = pred_prob[cols, candidates_idxs]
    costs = -np.sum(np.log(corr_prob + 1e-6), axis=1)
    return costs


def get_args():
    parser = argparse.ArgumentParser(description="BDD-OIA example")
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
        "--lr", type=float, default=2e-3, help="base model learning rate (default : 0.002)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="base model batch size (default : 32)"
    )
    parser.add_argument(
        "--loops", type=int, default=2, help="number of loop iterations (default : 2)"
    )
    parser.add_argument(
        "--segment_size", type=int, default=0.01, help="segment size (default : 0.01)"
    )
    parser.add_argument("--save_interval", type=int, default=1, help="save interval (default : 1)")
    parser.add_argument(
        "--max-revision", type=int, default=3, help="maximum revision in reasoner (default : 3)"
    )
    parser.add_argument(
        "--require-more-revision",
        type=int,
        default=3,
        help="require more revision in reasoner (default : 3)",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Build logger
    print_log("Abductive Learning on the BDD-OIA example.", logger="current")

    # -- Working with Data ------------------------------
    print_log("Working with Data.", logger="current")
    train_data = get_dataset(fname="train.npz", get_pseudo_label=True)
    val_data = get_dataset(fname="val.npz", get_pseudo_label=True)
    test_data = get_dataset(fname="test.npz", get_pseudo_label=True)

    # -- Building the Learning Part ---------------------
    print_log("Building the Learning Part.", logger="current")

    # Build necessary components for BDDNN
    cls = ConceptNet()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(cls.parameters(), lr=args.lr)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        pct_start=0.15,
        epochs=args.loops,
        steps_per_epoch=int(1 / args.segment_size) + 1,
    )

    # Build BDDNN
    base_model = BDDNN(
        cls,
        loss_fn,
        optimizer,
        scheduler=scheduler,
        device=device,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )

    # Build ABLModel
    model = BDDABLModel(base_model)

    # -- Building the Reasoning Part --------------------
    print_log("Building the Reasoning Part.", logger="current")

    # Build knowledge base
    kb = BDDKB()

    # Create reasoner
    reasoner = Reasoner(
        kb,
        dist_func=multi_label_confidence_dist,
        max_revision=args.max_revision,
        require_more_revision=args.require_more_revision,
    )

    # -- Building Evaluation Metrics --------------------
    print_log("Building Evaluation Metrics.", logger="current")
    metric_list = [SymbolAccuracy(prefix="bdd_oia"), BDDReasoningMetric(kb=kb, prefix="bdd_oia")]

    # -- Bridging Learning and Reasoning ----------------
    print_log("Bridge Learning and Reasoning.", logger="current")
    bridge = BDDBridge(model, reasoner, metric_list)

    # Retrieve the directory of the Log file and define the directory for saving the model weights.
    log_dir = ABLLogger.get_current_instance().log_dir
    weights_dir = osp.join(log_dir, "weights")

    #  Train and Test
    bridge.train(
        train_data=train_data,
        val_data=val_data,
        loops=args.loops,
        segment_size=args.segment_size,
        save_interval=args.save_interval,
        save_dir=weights_dir,
    )
    bridge.test(test_data)


if __name__ == "__main__":
    main()
