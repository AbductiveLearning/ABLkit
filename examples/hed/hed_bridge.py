import os
from collections import defaultdict
from typing import Any, List
import torch
from torch.utils.data import DataLoader

from abl.reasoning import ReasonerBase
from abl.learning import ABLModel, BasicNN
from abl.bridge import SimpleBridge
from abl.evaluation import BaseMetric
from abl.dataset import BridgeDataset, RegressionDataset
from abl.structures import ListData
from abl.utils import print_log

from examples.hed.utils import gen_mappings, InfiniteSampler
from examples.models.nn import SymbolNetAutoencoder
from examples.hed.datasets.get_hed import get_pretrain_data


class HEDBridge(SimpleBridge):
    def __init__(
        self,
        model: ABLModel,
        reasoner: ReasonerBase,
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
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.RMSprop(
                cls_autoencoder.parameters(), lr=0.001, alpha=0.9, weight_decay=1e-6
            )

            pretrain_model = BasicNN(
                cls_autoencoder,
                criterion,
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

            min_loss = pretrain_model.fit(pretrain_data_loader)
            print_log(f"min loss is {min_loss}", logger="current")
            save_parma_dic = {
                "model": cls_autoencoder.base_model.state_dict(),
            }

            torch.save(save_parma_dic, os.path.join(weights_dir, "pretrain_weights.pth"))

        self.model.load(load_path=os.path.join(weights_dir, "pretrain_weights.pth"))

    def abduce_pseudo_label(self, data_samples: ListData):
        candidate_mappings = gen_mappings([0, 1, 2, 3], ["+", "=", 0, 1])
        mapping_score = []
        abduced_pseudo_label_list = []
        for _mapping in candidate_mappings:
            self.reasoner.mapping = _mapping
            self.reasoner.remapping = dict(zip(_mapping.values(), _mapping.keys()))
            self.idx_to_pseudo_label(data_samples)
            abduced_pseudo_label = self.reasoner.abduce(data_samples)
            mapping_score.append(len(abduced_pseudo_label) - abduced_pseudo_label.count([]))
            abduced_pseudo_label_list.append(abduced_pseudo_label)

        max_revisible_instances = max(mapping_score)
        return_idx = mapping_score.index(max_revisible_instances)
        self.reasoner.mapping = candidate_mappings[return_idx]
        self.reasoner.mapping = dict(
            zip(self.reasoner.mapping.values(), self.reasoner.mapping.keys())
        )
        data_samples.abduced_pseudo_label = abduced_pseudo_label_list[return_idx]

        return data_samples.abduced_pseudo_label

    def check_training_impact(self, filtered_data_samples, data_samples):
        character_accuracy = self.model.valid(filtered_data_samples)
        revisible_ratio = len(filtered_data_samples.X) / len(data_samples.X)
        print_log(
            f"Revisible ratio is {revisible_ratio:.3f}, Character accuracy is {character_accuracy:.3f}",
            logger="current",
        )

        if character_accuracy >= 0.9 and revisible_ratio >= 0.9:
            return True
        return False

    def check_rule_quality(self, rule, val_data, equation_len):
        val_X_true = val_data[1][equation_len] + val_data[1][equation_len + 1]
        val_X_false = val_data[0][equation_len] + val_data[0][equation_len + 1]

        true_ratio = self.calc_consistent_ratio(val_X_true, rule)
        false_ratio = self.calc_consistent_ratio(val_X_false, rule)

        print_log(
            f"True consistent ratio is {true_ratio:.3f}, False inconsistent ratio is {1 - false_ratio:.3f}",
            logger="current",
        )

        if true_ratio > 0.95 and false_ratio < 0.1:
            return True
        return False

    def calc_consistent_ratio(self, X, rule):
        pred_label, _ = self.predict(X)
        pred_pseudo_label = self.label_to_pseudo_label(pred_label)
        consistent_num = sum(
            [self.reasoner.kb.consist_rule(instance, rule) for instance in pred_pseudo_label]
        )
        return consistent_num / len(X)

    def get_rules_from_data(self, train_data, samples_per_rule, samples_num):
        rules = []
        sampler = InfiniteSampler(len(train_data))
        data_loader = DataLoader(
            train_data,
            sampler=sampler,
            batch_size=samples_per_rule,
            collate_fn=lambda data_list: [list(data) for data in zip(*data_list)],
        )

        for _ in range(samples_num):
            for X, Y, Z in data_loader:
                pred_label, _ = self.predict(X)
                pred_pseudo_label = self.label_to_pseudo_label(pred_label)
                consistent_instance = []
                for instance in pred_pseudo_label:
                    if self.reasoner.kb.logic_forward([instance]):
                        consistent_instance.append(instance)

                if len(consistent_instance) != 0:
                    rule = self.reasoner.abduce_rules(consistent_instance)
                    if rule != None:
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
    def filter_empty(data_samples: ListData):
        consistent_dix = [
            i
            for i in range(len(data_samples.abduced_pseudo_label))
            if len(data_samples.abduced_pseudo_label[i]) > 0
        ]
        return data_samples[consistent_dix]

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

    def idx_to_pseudo_label(self, data_samples: ListData) -> List[List[Any]]:
        return super().idx_to_pseudo_label(data_samples)

    def train(self, train_data, val_data, segment_size=10, min_len=5, max_len=8):
        for equation_len in range(min_len, max_len):
            print_log(
                f"============== equation_len: {equation_len}-{equation_len + 1} ================",
                logger="current",
            )

            X = train_data[1][equation_len] + train_data[1][equation_len + 1]
            Y = [None] * len(X)
            data_samples = self.data_preprocess(X, None, Y)
            sampler = InfiniteSampler(len(data_samples))
            for seg_idx, select_idx in enumerate(sampler):
                sub_data_samples = data_samples[select_idx]
                self.predict(sub_data_samples)
                # self.idx_to_pseudo_label(sub_data_samples)
                self.abduce_pseudo_label(sub_data_samples)
                filtered_sub_data_samples = self.filter_empty(sub_data_samples)
                self.pseudo_label_to_idx(filtered_sub_data_samples)
                loss = self.model.train(filtered_sub_data_samples)

                print_log(
                    f"Equation Len(train) [{equation_len}] Segment Index [{seg_idx + 1}] model loss is {loss:.5f}",
                    logger="current",
                )

                if self.check_training_impact(filtered_sub_data_samples, sub_data_samples):
                    condition_num += 1
                else:
                    condition_num = 0

                if condition_num >= 5:
                    print_log(f"Now checking if we can go to next course", logger="current")
                    rules = self.get_rules_from_data(
                        data_samples, samples_per_rule=3, samples_num=50
                    )
                    print_log(f"Learned rules from data: " + str(rules), logger="current")

                    seems_good = self.check_rule_quality(rules, val_data, equation_len)
                    if seems_good:
                        self.model.save(save_path=f"./weights/eq_len_{equation_len}.pth")
                        break
                    else:
                        if equation_len == min_len:
                            print_log(
                                "Learned mapping is: " + str(self.reasoner.mapping),
                                logger="current",
                            )
                            self.model.load(load_path="./weights/pretrain_weights.pth")
                        else:
                            self.model.load(load_path=f"./weights/eq_len_{equation_len - 1}.pth")
                        condition_num = 0
                        print_log("Reload Model and retrain", logger="current")

    # def train(
    #     self,
    #     train_data,
    #     val_data,
    #     segment_size=10,
    #     min_len=5,
    #     max_len=8,
    # ):
    #     for equation_len in range(min_len, max_len):
    #         print_log(
    #             f"============== equation_len: {equation_len}-{equation_len + 1} ================",
    #             logger="current",
    #         )

    #         train_X = train_data[1][equation_len] + train_data[1][equation_len + 1]
    #         train_Y = [None] * len(train_X)
    #         # data_samples = self.data_preprocess(train_X, None, train_Y)

    #         dataset = BridgeDataset(train_X, None, train_Y)
    #         sampler = InfiniteSampler(len(dataset))
    #         data_loader = DataLoader(
    #             dataset,
    #             sampler=sampler,
    #             batch_size=segment_size,
    #             collate_fn=lambda data_list: [list(data) for data in zip(*data_list)],
    #         )

    #         condition_num = 0

    #         for seg_idx, (X, Z, Y) in enumerate(data_loader):
    #             pred_label, pred_prob = self.predict(ListData(X=X))
    #             if equation_len == min_len:
    #                 abduced_pseudo_label = self.select_mapping_and_abduce(pred_label, pred_prob, Y)
    #             else:
    #                 pred_pseudo_label = self.label_to_pseudo_label(pred_label)
    #                 abduced_pseudo_label = self.abduce_pseudo_label(
    #                     pred_label, pred_prob, pred_pseudo_label, Y, 20
    #                 )
    #             filtered_X, filtered_abduced_pseudo_label = self.filter_empty(
    #                 X, abduced_pseudo_label
    #             )
    #             if len(filtered_X) == 0:
    #                 continue
    #             filtered_abduced_label = self.pseudo_label_to_label(filtered_abduced_pseudo_label)
    #             min_loss = self.model.train(filtered_X, filtered_abduced_label)

    #             print_log(
    #                 f"Equation Len(train) [{equation_len}] Segment Index [{seg_idx + 1}] minimal_loss is {min_loss:.5f}",
    #                 logger="current",
    #             )

    #             if self.check_training_impact(filtered_X, filtered_abduced_label, X):
    #                 condition_num += 1
    #             else:
    #                 condition_num = 0

    #             if condition_num >= 5:
    #                 print_log(f"Now checking if we can go to next course", logger="current")
    #                 rules = self.get_rules_from_data(dataset, samples_per_rule=3, samples_num=50)
    #                 print_log(f"Learned rules from data: " + str(rules), logger="current")

    #                 seems_good = self.check_rule_quality(rules, val_data, equation_len)
    #                 if seems_good:
    #                     self.model.save(save_path=f"./weights/eq_len_{equation_len}.pth")
    #                     break
    #                 else:
    #                     if equation_len == min_len:
    #                         print_log(
    #                             "Learned mapping is: " + str(self.reasoner.mapping),
    #                             logger="current",
    #                         )
    #                         self.model.load(load_path="./weights/pretrain_weights.pth")
    #                     else:
    #                         self.model.load(load_path=f"./weights/eq_len_{equation_len - 1}.pth")
    #                     condition_num = 0
    #                     print_log("Reload Model and retrain", logger="current")
