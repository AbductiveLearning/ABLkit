from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np

from ..structures import ListData


class BaseKB(ABC):
    @abstractmethod
    def logic_forward(self, data_sample: ListData):
        """Placeholder for the forward reasoning of the knowledge base."""
        pass

    @abstractmethod
    def abduce_candidates(self, data_sample: ListData):
        """Placeholder for abduction of the knowledge base."""
        pass
