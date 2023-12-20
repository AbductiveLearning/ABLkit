# %%
import torch
import numpy as np
import torch.nn as nn
import os.path as osp

from abl.reasoning import Reasoner, KBBase
from abl.learning import BasicNN, ABLModel
from abl.bridge import SimpleBridge
from abl.evaluation import SymbolMetric, ReasoningMetric
from abl.utils import ABLLogger, print_log

from examples.models.nn import SymbolNet
from examples.hwf.datasets.get_dataset import get_hwf

# %%
# Initialize logger and print basic information
print_log("Abductive Learning on the HWF example.", logger="current")

# Retrieve the directory of the Log file and define the directory for saving the model weights.
log_dir = ABLLogger.get_current_instance().log_dir
weights_dir = osp.join(log_dir, "weights")

# %% [markdown]
# ### Logic Part

# %%
# Initialize knowledge base and reasoner
class HWF_KB(KBBase):
    def _valid_candidate(self, formula):
        if len(formula) % 2 == 0:
            return False
        for i in range(len(formula)):
            if i % 2 == 0 and formula[i] not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                return False
            if i % 2 != 0 and formula[i] not in ["+", "-", "*", "/"]:
                return False
        return True

    def logic_forward(self, formula):
        if not self._valid_candidate(formula):
            return np.info
        return eval("".join(formula))


kb = HWF_KB(
    pseudo_label_list=["1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/"],
    max_err=1e-10,
    use_cache=False,
)
reasoner = Reasoner(kb, dist_func="confidence")

# %% [markdown]
# ### Machine Learning Part

# %%
# Initialize necessary component for machine learning part
cls = SymbolNet(num_classes=len(kb.pseudo_label_list), image_size=(45, 45, 1))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cls.parameters(), lr=0.001, betas=(0.9, 0.99))

# %%
# Initialize BasicNN
# The function of BasicNN is to wrap NN models into the form of an sklearn estimator
base_model = BasicNN(
    model=cls,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
    save_interval=1,
    save_dir=weights_dir,
    batch_size=128,
    num_epochs=3,
)

# %%
# Initialize ABL model
# The main function of the ABL model is to serialize data and
# provide a unified interface for different machine learning models
model = ABLModel(base_model)

# %% [markdown]
# ### Metric

# %%
# Add metric
metric_list = [SymbolMetric(prefix="hwf"), ReasoningMetric(kb=kb, prefix="hwf")]

# %% [markdown]
# ### Dataset

# %%
# Get training and testing data
train_data = get_hwf(train=True, get_pseudo_label=True)
test_data = get_hwf(train=False, get_pseudo_label=True)

# %% [markdown]
# ### Bridge Machine Learning and Logic Reasoning

# %%
bridge = SimpleBridge(model=model, reasoner=reasoner, metric_list=metric_list)

# %% [markdown]
# ### Train and Test

# %%
bridge.train(train_data, train_data, loops=3, segment_size=1000, save_interval=1, save_dir=weights_dir)
bridge.test(test_data)


