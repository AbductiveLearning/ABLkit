from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy

from ...structures import ListData


class BaseSearchEngine(ABC):
    @abstractmethod
    def generator(data_sample: ListData) -> Union[List, Tuple, numpy.ndarray]:
        """Placeholder for the generator of revision_idx."""
        pass
