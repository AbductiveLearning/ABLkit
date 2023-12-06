import os.path as osp

import numpy as np
import torch
import torch.nn as nn

from abl.evaluation import SemanticsMetric, SymbolMetric
from abl.learning import ABLModel, BasicNN
from abl.reasoning import PrologKB, ReasonerBase
from abl.utils import ABLLogger, print_log, reform_list
from examples.hed.datasets.get_hed import get_hed, split_equation
from examples.hed.hed_bridge import HEDBridge
from examples.models.nn import SymbolNet

# Build logger
print_log("Abductive Learning on the HED example.", logger="current")

# Retrieve the directory of the Log file and define the directory for saving the model weights.
log_dir = ABLLogger.get_current_instance().log_dir
weights_dir = osp.join(log_dir, "weights")


### Logic Part
# Initialize knowledge base and abducer
class HedKB(PrologKB):
    def __init__(self, pseudo_label_list, pl_file):
        super().__init__(pseudo_label_list, pl_file)

    def consist_rule(self, exs, rules):
        rules = str(rules).replace("'", "")
        return len(list(self.prolog.query("eval_inst_feature(%s, %s)." % (exs, rules)))) != 0

    def abduce_rules(self, pred_res):
        prolog_result = list(self.prolog.query("consistent_inst_feature(%s, X)." % pred_res))
        if len(prolog_result) == 0:
            return None
        prolog_rules = prolog_result[0]["X"]
        rules = [rule.value for rule in prolog_rules]
        return rules


class HedReasoner(ReasonerBase):
    def revise_at_idx(self, data_sample):
        revision_idx = np.where(np.array(data_sample.flatten("revision_flag")) != 0)[0]
        candidate = self.kb.revise_at_idx(
            data_sample.pred_pseudo_label, data_sample.Y, revision_idx
        )
        return candidate

    def zoopt_revision_score(self, symbol_num, data_sample, sol):
        revision_flag = reform_list(list(sol.get_x().astype(np.int32)), data_sample.pred_pseudo_label)
        data_sample.revision_flag = revision_flag

        lefted_idxs = [i for i in range(len(data_sample.pred_idx))]
        candidate_size = []
        while lefted_idxs:
            idxs = []
            idxs.append(lefted_idxs.pop(0))
            max_candidate_idxs = []
            found = False
            for idx in range(-1, len(data_sample.pred_idx)):
                if (not idx in idxs) and (idx >= 0):
                    idxs.append(idx)
                candidate = self.revise_at_idx(data_sample[idxs])
                if len(candidate) == 0:
                    if len(idxs) > 1:
                        idxs.pop()
                else:
                    if len(idxs) > len(max_candidate_idxs):
                        found = True
                        max_candidate_idxs = idxs.copy()
            removed = [i for i in lefted_idxs if i in max_candidate_idxs]
            if found:
                candidate_size.append(len(removed) + 1)
                lefted_idxs = [i for i in lefted_idxs if i not in max_candidate_idxs]
        candidate_size.sort()
        score = 0
        import math

        for i in range(0, len(candidate_size)):
            score -= math.exp(-i) * candidate_size[i]
        return score

    def abduce(self, data_sample):
        symbol_num = data_sample.elements_num("pred_pseudo_label")
        max_revision_num = self._get_max_revision_num(self.max_revision, symbol_num)

        solution = self.zoopt_get_solution(symbol_num, data_sample, max_revision_num)

        data_sample.revision_flag = reform_list(
            solution.astype(np.int32), data_sample.pred_pseudo_label
        )

        abduced_pseudo_label = []

        for single_instance in data_sample:
            single_instance.pred_pseudo_label = [single_instance.pred_pseudo_label]
            candidates = self.revise_at_idx(single_instance)
            if len(candidates) == 0:
                abduced_pseudo_label.append([])
            else:
                abduced_pseudo_label.append(candidates[0][0])
        data_sample.abduced_pseudo_label = abduced_pseudo_label
        return abduced_pseudo_label

    def abduce_rules(self, pred_res):
        return self.kb.abduce_rules(pred_res)


import os

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

kb = HedKB(
    pseudo_label_list=[1, 0, "+", "="], pl_file=os.path.join(CURRENT_DIR, "./datasets/learn_add.pl")
)
reasoner = HedReasoner(kb, dist_func="hamming", use_zoopt=True, max_revision=20)

### Machine Learning Part
# Build necessary components for BasicNN
cls = SymbolNet(num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cls.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Build BasicNN
# The function of BasicNN is to wrap NN models into the form of an sklearn estimator
base_model = BasicNN(
    cls,
    criterion,
    optimizer,
    device,
    batch_size=32,
    num_epochs=1,
    save_interval=1,
    save_dir=weights_dir,
)

# Build ABLModel
# The main function of the ABL model is to serialize data and
# provide a unified interface for different machine learning models
model = ABLModel(base_model)

### Metric
# Set up metrics
metric_list = [SymbolMetric(prefix="hed"), SemanticsMetric(prefix="hed")]

### Bridge Machine Learning and Logic Reasoning
bridge = HEDBridge(model, reasoner, metric_list)

### Dataset
total_train_data = get_hed(train=True)
train_data, val_data = split_equation(total_train_data, 3, 1)
test_data = get_hed(train=False)

### Train and Test
bridge.pretrain("examples/hed/weights")
bridge.train(train_data, val_data)
