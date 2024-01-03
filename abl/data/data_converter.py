from typing import Any, Tuple

from abl.utils import tab_data_to_tuple
from .structures.list_data import ListData
from lambdaLearn.Base.TabularMixin import TabularMixin

class DataConverter:
    '''
    This class provides functionality to convert LambdaLearn data to ABL-Package data.
    '''
    def __init__(self) -> None:
        pass
    
    def convert_lambdalearn_to_tuple(
        self,
        dataset: TabularMixin,
        reasoning_result: Any
    ) -> Tuple[Tuple, Tuple, Tuple, Tuple]:
        '''
        Convert a lambdalearn dataset to a tuple of tuples (label_data, train_data, valid_data, test_data), each containing (data, label, reasoning_result).
        
        Parameters
        ----------
        dataset : TabularMixin
            The LambdaLearn dataset to be converted.
        reasoning_result : Any
            The reasoning result of the dataset.
        Returns
        -------
        Tuple[Tuple, Tuple, Tuple, Tuple]
            A tuple of (label_data, train_data, valid_data, test_data), where each element is a tuple of (data, label, reasoning_result).
        '''
        
        if not isinstance(dataset, TabularMixin):
            raise NotImplementedError("Only support converting the datasets that are instances of TabularMixin. Please refer to the documentation and manually convert the dataset into a tuple. ")
        
        label_data = tab_data_to_tuple(dataset.labeled_X, dataset.labeled_y, reasoning_result=reasoning_result)
        train_data = tab_data_to_tuple(dataset.unlabeled_X, dataset.unlabeled_y, reasoning_result=reasoning_result)
        valid_data = tab_data_to_tuple(dataset.valid_X, dataset.valid_y, reasoning_result=reasoning_result)
        test_data = tab_data_to_tuple(dataset.test_X, dataset.test_y, reasoning_result=reasoning_result)
        
        return label_data, train_data, valid_data, test_data
    
    def convert_lambdalearn_to_listdata(
        self,
        dataset: TabularMixin,
        reasoning_result: Any
    ) -> Tuple[ListData, ListData, ListData, ListData]:
        '''
        Convert a lambdalearn dataset to a tuple of ListData (label_data_examples, train_data_examples, valid_data_examples, test_data_examples).
        
        Parameters
        ----------
        dataset : TabularMixin
            The LambdaLearn dataset to be converted.
        reasoning_result : Any
            The reasoning result of the dataset.
        Returns
        -------
        Tuple[ListData, ListData, ListData, ListData]
            A tuple of ListData (label_data_examples, train_data_examples, valid_data_examples, test_data_examples)
        '''
        
        if not isinstance(dataset, TabularMixin):
            raise NotImplementedError("Only support converting the datasets that are instances of TabularMixin. Please refer to the documentation and manually convert the dataset into a ListData. ")
        
        label_data, train_data, valid_data, test_data = self.convert_lambdalearn_to_tuple(dataset, reasoning_result)
        
        if label_data is not None:
            X, gt_pseudo_label, Y = label_data
            label_data_examples = ListData(X=X, gt_pseudo_label=gt_pseudo_label, Y=Y)
        if train_data is not None:
            X, gt_pseudo_label, Y = train_data
            train_data_examples = ListData(X=X, gt_pseudo_label=gt_pseudo_label, Y=Y)
        if valid_data is not None:
            X, gt_pseudo_label, Y = valid_data
            valid_data_examples = ListData(X=X, gt_pseudo_label=gt_pseudo_label, Y=Y)
        if test_data is not None:
            X, gt_pseudo_label, Y = test_data
            test_data_examples = ListData(X=X, gt_pseudo_label=gt_pseudo_label, Y=Y)
        
        return label_data_examples, train_data_examples, valid_data_examples, test_data_examples