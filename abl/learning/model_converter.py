import torch
import copy
from typing import Any, Callable, List, Optional

from .abl_model import ABLModel
from .basic_nn import BasicNN
from lambdaLearn.Base.DeepModelMixin import DeepModelMixin


class ModelConverter:
    """
    This class provides functionality to convert LambdaLearn models to ABL-Package models.
    """

    def __init__(self) -> None:
        pass

    def convert_lambdalearn_to_ablmodel(
        self,
        lambdalearn_model,
        loss_fn: torch.nn.Module,
        optimizer_dict: dict,
        scheduler_dict: Optional[dict] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        num_epochs: int = 1,
        stop_loss: Optional[float] = 0.0001,
        num_workers: int = 0,
        save_interval: Optional[int] = None,
        save_dir: Optional[str] = None,
        train_transform: Callable[..., Any] = None,
        test_transform: Callable[..., Any] = None,
        collate_fn: Callable[[List[Any]], Any] = None,
    ):
        """
        Convert a lambdalearn model to an ABLModel. If the lambdalearn model is an instance of
        DeepModelMixin, its network will be used as the model of BasicNN. Otherwise, the lambdalearn
        model should implement ``fit`` and ``predict`` methods.

        Parameters
        ----------
        lambdalearn_model : Union[DeepModelMixin, Any]
            The LambdaLearn model to be converted.
        loss_fn : torch.nn.Module
            The loss function used for training.
        optimizer_dict : dict
            The dict contains necessary parameters to construct a optimizer used for training.
            The optimizer class is specified by the ``optimizer`` key.
        scheduler_dict : dict, optional
            The dict contains necessary parameters to construct a learning rate scheduler used
            for training, which will be called at the end of each run of the ``fit`` method.
            The scheduler class is specified by the ``scheduler`` key. It should implement the
            ``step`` method, by default None.
        device : torch.device, optional
            The device on which the model will be trained or used for prediction,
            by default torch.device("cpu").
        batch_size : int, optional
            The batch size used for training, by default 32.
        num_epochs : int, optional
            The number of epochs used for training, by default 1.
        stop_loss : float, optional
            The loss value at which to stop training, by default 0.0001.
        num_workers : int
            The number of workers used for loading data, by default 0.
        save_interval : int, optional
            The model will be saved every ``save_interval`` epoch during training, by default None.
        save_dir : str, optional
            The directory in which to save the model during training, by default None.
        train_transform : Callable[..., Any], optional
            A function/transform that takes an object and returns a transformed version used
            in the `fit` and `train_epoch` methods, by default None.
        test_transform : Callable[..., Any], optional
            A function/transform that takes an object and returns a transformed version in the
            `predict`, `predict_proba` and `score` methods, , by default None.
        collate_fn : Callable[[List[T]], Any], optional
            The function used to collate data, by default None.

        Returns
        -------
        ABLModel
            The converted ABLModel instance.
        """
        if isinstance(lambdalearn_model, DeepModelMixin):
            base_model = self.convert_lambdalearn_to_basicnn(
                lambdalearn_model,
                loss_fn,
                optimizer_dict,
                scheduler_dict,
                device,
                batch_size,
                num_epochs,
                stop_loss,
                num_workers,
                save_interval,
                save_dir,
                train_transform,
                test_transform,
                collate_fn,
            )
            return ABLModel(base_model)

        if not (hasattr(lambdalearn_model, "fit") and hasattr(lambdalearn_model, "predict")):
            raise NotImplementedError(
                "The lambdalearn_model should be an instance of DeepModelMixin, or implement "
                + "fit and predict methods."
            )

        return ABLModel(lambdalearn_model)

    def convert_lambdalearn_to_basicnn(
        self,
        lambdalearn_model: DeepModelMixin,
        loss_fn: torch.nn.Module,
        optimizer_dict: dict,
        scheduler_dict: Optional[dict] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        num_epochs: int = 1,
        stop_loss: Optional[float] = 0.0001,
        num_workers: int = 0,
        save_interval: Optional[int] = None,
        save_dir: Optional[str] = None,
        train_transform: Callable[..., Any] = None,
        test_transform: Callable[..., Any] = None,
        collate_fn: Callable[[List[Any]], Any] = None,
    ):
        """
        Convert a lambdalearn model to a BasicNN. If the lambdalearn model is an instance of
        DeepModelMixin, its network will be used as the model of BasicNN.

        Parameters
        ----------
        lambdalearn_model : Union[DeepModelMixin, Any]
            The LambdaLearn model to be converted.
        loss_fn : torch.nn.Module
            The loss function used for training.
        optimizer_dict : dict
            The dict contains necessary parameters to construct a optimizer used for training.
        scheduler_dict : dict, optional
            The dict contains necessary parameters to construct a learning rate scheduler used
            for training, which will be called at the end of each run of the ``fit`` method.
            The scheduler class is specified by the ``scheduler`` key. It should implement the
            ``step`` method, by default None.
        device : torch.device, optional
            The device on which the model will be trained or used for prediction,
            by default torch.device("cpu").
        batch_size : int, optional
            The batch size used for training, by default 32.
        num_epochs : int, optional
            The number of epochs used for training, by default 1.
        stop_loss : float, optional
            The loss value at which to stop training, by default 0.0001.
        num_workers : int
            The number of workers used for loading data, by default 0.
        save_interval : int, optional
            The model will be saved every ``save_interval`` epoch during training, by default None.
        save_dir : str, optional
            The directory in which to save the model during training, by default None.
        train_transform : Callable[..., Any], optional
            A function/transform that takes an object and returns a transformed version used
            in the `fit` and `train_epoch` methods, by default None.
        test_transform : Callable[..., Any], optional
            A function/transform that takes an object and returns a transformed version in the
            `predict`, `predict_proba` and `score` methods, , by default None.
        collate_fn : Callable[[List[T]], Any], optional
            The function used to collate data, by default None.

        Returns
        -------
        BasicNN
            The converted BasicNN instance.
        """
        if isinstance(lambdalearn_model, DeepModelMixin):
            if not isinstance(lambdalearn_model.network, torch.nn.Module):
                raise NotImplementedError(
                    "Expected lambdalearn_model.network to be a torch.nn.Module, "
                    + f"but got {type(lambdalearn_model.network)}"
                )
            # Only use the network part and device of the lambdalearn model
            network = copy.deepcopy(lambdalearn_model.network)
            optimizer_class = optimizer_dict["optimizer"]
            optimizer_dict.pop("optimizer")
            optimizer = optimizer_class(network.parameters(), **optimizer_dict)
            if scheduler_dict is not None:
                scheduler_class = scheduler_dict["scheduler"]
                scheduler_dict.pop("scheduler")
                scheduler = scheduler_class(optimizer, **scheduler_dict)
            else:
                scheduler = None
            device = lambdalearn_model.device if device is None else device
            base_model = BasicNN(
                model=network,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                batch_size=batch_size,
                num_epochs=num_epochs,
                stop_loss=stop_loss,
                num_workers=num_workers,
                save_interval=save_interval,
                save_dir=save_dir,
                train_transform=train_transform,
                test_transform=test_transform,
                collate_fn=collate_fn,
            )
            return base_model
        else:
            raise NotImplementedError(
                "The lambdalearn_model should be an instance of DeepModelMixin."
            )
