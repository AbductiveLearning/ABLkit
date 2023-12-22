import pickle
from typing import Any, Dict

from ..data.structures import ListData
from ..utils import reform_list


class ABLModel:
    """
    Serialize data and provide a unified interface for different machine learning models.

    Parameters
    ----------
    base_model : Machine Learning Model
        The machine learning base model used for training and prediction. This model should
        implement the ``fit`` and ``predict`` methods. It's recommended, but not required, for the
        model to also implement the ``predict_proba`` method for generating
        predictions on the probabilities.
    """

    def __init__(self, base_model: Any) -> None:
        if not (hasattr(base_model, "fit") and hasattr(base_model, "predict")):
            raise NotImplementedError("The base_model should implement fit and predict methods.")

        self.base_model = base_model

    def predict(self, data_examples: ListData) -> Dict:
        """
        Predict the labels and probabilities for the given data.

        Parameters
        ----------
        data_examples : ListData
            A batch of data to predict on.

        Returns
        -------
        dict
            A dictionary containing the predicted labels and probabilities.
        """
        model = self.base_model
        data_X = data_examples.flatten("X")
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X=data_X)
            label = prob.argmax(axis=1)
            prob = reform_list(prob, data_examples.X)
        else:
            prob = None
            label = model.predict(X=data_X)
        label = reform_list(label, data_examples.X)

        data_examples.pred_idx = label
        data_examples.pred_prob = prob

        return {"label": label, "prob": prob}

    def train(self, data_examples: ListData) -> float:
        """
        Train the model on the given data.

        Parameters
        ----------
        data_examples : ListData
            A batch of data to train on, which typically contains the data, ``X``, and the
            corresponding labels, ``abduced_idx``.

        Returns
        -------
        float
            The loss value of the trained model.
        """
        data_X = data_examples.flatten("X")
        data_y = data_examples.flatten("abduced_idx")
        return self.base_model.fit(X=data_X, y=data_y)

    def valid(self, data_examples: ListData) -> float:
        """
        Validate the model on the given data.

        Parameters
        ----------
        data_examples : ListData
            A batch of data to train on, which typically contains the data, ``X``,
            and the corresponding labels, ``abduced_idx``.

        Returns
        -------
        float
            The accuracy the trained model.
        """
        data_X = data_examples.flatten("X")
        data_y = data_examples.flatten("abduced_idx")
        score = self.base_model.score(X=data_X, y=data_y)
        return score

    def _model_operation(self, operation: str, *args, **kwargs):
        model = self.base_model
        if hasattr(model, operation):
            method = getattr(model, operation)
            method(*args, **kwargs)
        else:
            if f"{operation}_path" not in kwargs.keys():
                raise ValueError(f"'{operation}_path' should not be None")
            else:
                try:
                    if operation == "save":
                        with open(kwargs["save_path"], "wb") as file:
                            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
                    elif operation == "load":
                        with open(kwargs["load_path"], "rb") as file:
                            self.base_model = pickle.load(file)
                except (OSError, pickle.PickleError):
                    raise NotImplementedError(
                        f"{type(model).__name__} object doesn't have the {operation} method \
                            and the default pickle-based {operation} method failed."
                    )

    def save(self, *args, **kwargs) -> None:
        """
        Save the model to a file.

        This method delegates to the ``save`` method of self.base_model. The arguments passed to
        this method should match those expected by the ``save`` method of self.base_model.
        """
        self._model_operation("save", *args, **kwargs)

    def load(self, *args, **kwargs) -> None:
        """
        Load the model from a file.

        This method delegates to the ``load`` method of self.base_model. The arguments passed to
        this method should match those expected by the ``load`` method of self.base_model.
        """
        self._model_operation("load", *args, **kwargs)
