import numpy as np


class VotingRegressor:
    """
    Implements a voting regressor.
    Takes a list of fitted models and their weights and returns a weighted average of the predictions.
    """

    def __init__(self, models, model_weights=None):
        """
        Initialize the VotingRegressor object.

        Parameters:
        - models: list of models to be stacked
        - model_weights: list of weights for each model. Default is None.
        """
        self.models = models
        self.model_weights = model_weights

    def predict(self, X):
        """
        Predict the target variable using the fitted models.

        Parameters:
        - X: input features

        Returns:
        - y_pred: predicted target variable
        """
        y_preds = []
        for model in self.models:
            y_pred = model.predict(X)
            y_preds.append(y_pred)

        return np.average(y_preds, axis=0, weights=self.model_weights)

    def get_params(self):
        """
        Get the parameters of the VotingRegressor object.

        Returns:
        - params: dictionary of parameters
        """
        return {"models": self.models, "model_weights": self.model_weights}

    def show_models(self, formula=False):
        """
        Print the models and their weights.
        """
        for model, weight in zip(self.models, self.model_weights, strict=False):
            if formula:
                print(
                    f"Model: {model}, Weight: {weight} \n\tFormula: {model.get_formula()}"
                )
            else:
                print(f"Model: {model}, Weight: {weight}")
