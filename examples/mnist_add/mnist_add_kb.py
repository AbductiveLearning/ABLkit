from typing import Any

from abl.reasoning import SearchBasedKB
from abl.structures import ListData


class AddKB(SearchBasedKB):
    def __init__(self, pseudo_label_list=list(range(10))):
        super().__init__(
            pseudo_label_list=pseudo_label_list
        )

    def check_equal(self, data_sample: ListData, y: Any):
        return self.logic_forward(data_sample) == y

    def logic_forward(self, data_sample):
        return sum(data_sample["pred_pseudo_label"][0])
