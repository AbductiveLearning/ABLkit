import numpy as np
from typing import List, Any
from ablkit.data import ListData
from ablkit.bridge import SimpleBridge

class BDDBridge(SimpleBridge):
    def idx_to_pseudo_label(self, data_examples: ListData) -> List[List[Any]]:
        pred_idx = data_examples.pred_idx  # [ ndarray(1,nc),... ]
        pred_pseudo_label = []
        for sub_list in pred_idx:
            sub_list = sub_list.squeeze()  # 1 x nc -> nc
            pred_pseudo_label.append([self.reasoner.idx_to_label[_idx] for _idx in sub_list])
        data_examples.pred_pseudo_label = pred_pseudo_label
        return data_examples.pred_pseudo_label

    def pseudo_label_to_idx(self, data_examples: ListData) -> List[List[Any]]:
        abduced_pseudo_label = data_examples.abduced_pseudo_label
        abduced_idx = []
        for sub_list in abduced_pseudo_label:
            sub_list = np.array([self.reasoner.label_to_idx[_lab] for _lab in sub_list])
            abduced_idx.append(sub_list)
        data_examples.abduced_idx = abduced_idx
        return data_examples.abduced_idx