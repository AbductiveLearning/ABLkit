from typing import Dict

import numpy as np
from ablkit.data import ListData
from ablkit.learning import ABLModel
from ablkit.utils import reform_list


class BDDABLModel(ABLModel):
    def predict(self, data_examples: ListData) -> Dict:
        model = self.base_model
        data_X = data_examples.flatten("X")
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X=data_X)
            label = np.where(prob > 0.5, 1, 0).astype(int)
            prob = reform_list(prob, data_examples.X)
        else:
            prob = None
            label = model.predict(X=data_X)
        label = reform_list(label, data_examples.X)

        data_examples.pred_idx = label
        data_examples.pred_prob = prob

        return {"label": label, "prob": prob}
