import os
from collections import defaultdict

import torch

from abl.bridge import SimpleBridge
from abl.dataset import RegressionDataset
from abl.evaluation import BaseMetric
from abl.learning import ABLModel, BasicNN
from abl.reasoning import Reasoner
from abl.structures import ListData
from abl.utils import print_log
from examples.hed.datasets.get_dataset import get_pretrain_data
from examples.hed.utils import InfiniteSampler, gen_mappings
from examples.models.nn import SymbolNetAutoencoder


class HEDBridge(SimpleBridge):
    def __init__(
        self,
        model: ABLModel,
        reasoner: Reasoner,
        metric_list: BaseMetric,
    ) -> None:
        super().__init__(model, reasoner, metric_list)

    def pretrain(self, weights_dir):
        if not os.path.exists(os.path.join(weights_dir, "pretrain_weights.pth")):
            print_log("Pretrain Start", logger="current")

            cls_autoencoder = SymbolNetAutoencoder(
                num_classes=len(self.reasoner.kb.pseudo_label_list)
            )
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.RMSprop(
                cls_autoencoder.parameters(), lr=0.001, alpha=0.9, weight_decay=1e-6
            )

            pretrain_model = BasicNN(
                cls_autoencoder,
                loss_fn,
                optimizer,
                device,
                save_interval=1,
                save_dir=weights_dir,
                num_epochs=10,
            )

            pretrain_data_X, pretrain_data_Y = get_pretrain_data(["0", "1", "10", "11"])
            pretrain_data = RegressionDataset(pretrain_data_X, pretrain_data_Y)
            pretrain_data_loader = torch.utils.data.DataLoader(
                pretrain_data, batch_size=64, shuffle=True
            )

            pretrain_model.fit(pretrain_data_loader)
            save_parma_dic = {
                "model": cls_autoencoder.base_model.state_dict(),
            }

            torch.save(save_parma_dic, os.path.join(weights_dir, "pretrain_weights.pth"))

        self.model.load(load_path=os.path.join(weights_dir, "pretrain_weights.pth"))

    def select_mapping_and_abduce(self, data_examples: ListData):
        candidate_mappings = gen_mappings([0, 1, 2, 3], ["+", "=", 0, 1])
        mapping_score = []
        abduced_pseudo_label_list = []
        for _mapping in candidate_mappings:
            self.reasoner.idx_to_label = _mapping
            self.reasoner.label_to_idx = dict(zip(_mapping.values(), _mapping.keys()))
            self.idx_to_pseudo_label(data_examples)
            abduced_pseudo_label = self.reasoner.abduce(data_examples)
            mapping_score.append(len(abduced_pseudo_label) - abduced_pseudo_label.count([]))
            abduced_pseudo_label_list.append(abduced_pseudo_label)

        max_revisible_instances = max(mapping_score)
        return_idx = mapping_score.index(max_revisible_instances)
        self.reasoner.idx_to_label = candidate_mappings[return_idx]
        self.reasoner.label_to_idx = dict(
            zip(self.reasoner.idx_to_label.values(), self.reasoner.idx_to_label.keys())
        )
        self.idx_to_pseudo_label(data_examples)
        data_examples.abduced_pseudo_label = abduced_pseudo_label_list[return_idx]

        return data_examples.abduced_pseudo_label

    def abduce_pseudo_label(self, data_examples: ListData):
        self.reasoner.abduce(data_examples)
        return data_examples.abduced_pseudo_label

    def check_training_impact(self, filtered_data_examples, data_examples):
        character_accuracy = self.model.valid(filtered_data_examples)
        revisible_ratio = len(filtered_data_examples.X) / len(data_examples.X)
        log_string = (
            f"Revisible ratio is {revisible_ratio:.3f}, Character "
            f"accuracy is {character_accuracy:.3f}"
        )
        print_log(log_string, logger="current")

        if character_accuracy >= 0.95 and revisible_ratio >= 0.95:
            return True
        return False

    def check_rule_quality(self, rule, val_data, equation_len):
        val_X_true = self.data_preprocess(val_data[1], equation_len)
        val_X_false = self.data_preprocess(val_data[0], equation_len)

        true_ratio = self.calc_consistent_ratio(val_X_true, rule)
        false_ratio = self.calc_consistent_ratio(val_X_false, rule)

        log_string = (
            f"True consistent ratio is {true_ratio:.3f}, False inconsistent ratio "
            f"is {1 - false_ratio:.3f}"
        )
        print_log(log_string, logger="current")

        if true_ratio > 0.95 and false_ratio < 0.1:
            return True
        return False

    def calc_consistent_ratio(self, data_examples, rule):
        self.predict(data_examples)
        pred_pseudo_label = self.idx_to_pseudo_label(data_examples)
        consistent_num = sum(
            [self.reasoner.kb.consist_rule(instance, rule) for instance in pred_pseudo_label]
        )
        return consistent_num / len(data_examples.X)

    def get_rules_from_data(self, data_examples, samples_per_rule, samples_num):
        rules = []
        sampler = InfiniteSampler(len(data_examples), batch_size=samples_per_rule)

        for _ in range(samples_num):
            for select_idx in sampler:
                sub_data_examples = data_examples[select_idx]
                self.predict(sub_data_examples)
                pred_pseudo_label = self.idx_to_pseudo_label(sub_data_examples)
                consistent_instance = []
                for instance in pred_pseudo_label:
                    if self.reasoner.kb.logic_forward([instance]):
                        consistent_instance.append(instance)

                if len(consistent_instance) != 0:
                    rule = self.reasoner.abduce_rules(consistent_instance)
                    if rule is not None:
                        rules.append(rule)
                        break

        all_rule_dict = defaultdict(int)
        for rule in rules:
            for r in rule:
                all_rule_dict[r] += 1
        rule_dict = {rule: cnt for rule, cnt in all_rule_dict.items() if cnt >= 5}
        rules = self.select_rules(rule_dict)

        return rules

    @staticmethod
    def filter_empty(data_examples: ListData):
        consistent_dix = [
            i
            for i in range(len(data_examples.abduced_pseudo_label))
            if len(data_examples.abduced_pseudo_label[i]) > 0
        ]
        return data_examples[consistent_dix]

    @staticmethod
    def select_rules(rule_dict):
        add_nums_dict = {}
        for r in list(rule_dict):
            add_nums = str(r.split("]")[0].split("[")[1]) + str(
                r.split("]")[1].split("[")[1]
            )  # r = 'my_op([1], [0], [1, 0])' then add_nums = '10'
            if add_nums in add_nums_dict:
                old_r = add_nums_dict[add_nums]
                if rule_dict[r] >= rule_dict[old_r]:
                    rule_dict.pop(old_r)
                    add_nums_dict[add_nums] = r
                else:
                    rule_dict.pop(r)
            else:
                add_nums_dict[add_nums] = r
        return list(rule_dict)

    def data_preprocess(self, data, equation_len) -> ListData:
        data_examples = ListData()
        data_examples.X = data[equation_len] + data[equation_len + 1]
        data_examples.gt_pseudo_label = None
        data_examples.Y = [None] * len(data_examples.X)

        return data_examples

    def train(self, train_data, val_data, segment_size=10, min_len=5, max_len=8):
        for equation_len in range(min_len, max_len):
            print_log(
                f"============== equation_len: {equation_len}-{equation_len + 1} ================",
                logger="current",
            )

            condition_num = 0
            data_examples = self.data_preprocess(train_data[1], equation_len)
            sampler = InfiniteSampler(len(data_examples), batch_size=segment_size)
            for seg_idx, select_idx in enumerate(sampler):
                print_log(
                    f"Equation Len(train) [{equation_len}] Segment Index [{seg_idx + 1}]",
                    logger="current",
                )
                sub_data_examples = data_examples[select_idx]
                self.predict(sub_data_examples)
                if equation_len == min_len:
                    self.select_mapping_and_abduce(sub_data_examples)
                else:
                    self.idx_to_pseudo_label(sub_data_examples)
                    self.abduce_pseudo_label(sub_data_examples)
                filtered_sub_data_examples = self.filter_empty(sub_data_examples)
                self.pseudo_label_to_idx(filtered_sub_data_examples)
                loss = self.model.train(filtered_sub_data_examples)

                if self.check_training_impact(filtered_sub_data_examples, sub_data_examples):
                    condition_num += 1
                else:
                    condition_num = 0

                if condition_num >= 5:
                    print_log("Now checking if we can go to next course", logger="current")
                    rules = self.get_rules_from_data(
                        data_examples, samples_per_rule=3, samples_num=50
                    )
                    print_log("Learned rules from data: " + str(rules), logger="current")

                    seems_good = self.check_rule_quality(rules, val_data, equation_len)
                    if seems_good:
                        self.model.save(save_path=f"./weights/eq_len_{equation_len}.pth")
                        break
                    else:
                        if equation_len == min_len:
                            print_log(
                                "Learned mapping is: " + str(self.reasoner.idx_to_label),
                                logger="current",
                            )
                            self.model.load(load_path="./weights/pretrain_weights.pth")
                        else:
                            self.model.load(load_path=f"./weights/eq_len_{equation_len - 1}.pth")
                        condition_num = 0
                        print_log("Reload Model and retrain", logger="current")
