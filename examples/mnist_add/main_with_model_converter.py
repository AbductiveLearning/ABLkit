import argparse
import os.path as osp

from torch import nn
from torch.optim import RMSprop, lr_scheduler

from lambdaLearn.Algorithm.AbductiveLearning.bridge import SimpleBridge
from lambdaLearn.Algorithm.AbductiveLearning.data.evaluation import ReasoningMetric, SymbolAccuracy
from lambdaLearn.Algorithm.AbductiveLearning.learning import ABLModel
from lambdaLearn.Algorithm.AbductiveLearning.learning.model_converter import ModelConverter
from lambdaLearn.Algorithm.AbductiveLearning.reasoning import GroundKB, KBBase, PrologKB, Reasoner
from lambdaLearn.Algorithm.AbductiveLearning.utils import ABLLogger, print_log
from lambdaLearn.Algorithm.SemiSupervised.Classification.FixMatch import FixMatch

from datasets import get_dataset
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


def main():
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
        "--segment_size", type=int, default=0.01, help="segment size (default : 0.01)"
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
    kb_type = parser.add_mutually_exclusive_group()
    kb_type.add_argument(
        "--prolog", action="store_true", default=False, help="use PrologKB (default: False)"
    )
    kb_type.add_argument(
        "--ground", action="store_true", default=False, help="use GroundKB (default: False)"
    )

    args = parser.parse_args()

    # Build logger
    print_log("Abductive Learning on the MNIST Addition example.", logger="current")

    # -- Working with Data ------------------------------
    print_log("Working with Data.", logger="current")
    train_data = get_dataset(train=True, get_pseudo_label=True)
    test_data = get_dataset(train=False, get_pseudo_label=True)

    # -- Building the Learning Part ---------------------
    print_log("Building the Learning Part.", logger="current")

    # Build necessary components for BasicNN
    model = FixMatch(
        network=LeNet5(),
        threshold=0.95,
        lambda_u=1.0,
        mu=7,
        T=0.5,
        epoch=1,
        num_it_epoch=2**20,
        num_it_total=2**20,
        device="cuda",
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer_dict = dict(optimizer=RMSprop, lr=0.0003, alpha=0.9)
    scheduler_dict = dict(
        scheduler=lr_scheduler.OneCycleLR, max_lr=0.0003, pct_start=0.15, total_steps=200
    )

    converter = ModelConverter()
    base_model = converter.convert_lambdalearn_to_basicnn(
        model, loss_fn=loss_fn, optimizer_dict=optimizer_dict, scheduler_dict=scheduler_dict
    )

    # Build ABLModel
    model = ABLModel(base_model)

    # -- Building the Reasoning Part --------------------
    print_log("Building the Reasoning Part.", logger="current")

    # Build knowledge base
    if args.prolog:
        kb = PrologKB(pseudo_label_list=list(range(10)), pl_file="add.pl")
    elif args.ground:
        kb = AddGroundKB()
    else:
        kb = AddKB()

    # Create reasoner
    reasoner = Reasoner(
        kb, max_revision=args.max_revision, require_more_revision=args.require_more_revision
    )

    # -- Building Evaluation Metrics --------------------
    print_log("Building Evaluation Metrics.", logger="current")
    metric_list = [SymbolAccuracy(prefix="mnist_add"), ReasoningMetric(kb=kb, prefix="mnist_add")]

    # -- Bridging Learning and Reasoning ----------------
    print_log("Bridge Learning and Reasoning.", logger="current")
    bridge = SimpleBridge(model, reasoner, metric_list)

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
