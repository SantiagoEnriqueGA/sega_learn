import numpy as np

class Metrics(object):
    # Regression Metrics
    # TODO: Implement other regression metrics
    def mean_squared_error(self, y_true, y_pred):
        """
        Calculates the mean squared error between the true and predicted values.
        
        Parameters:
        - y_true (numpy.ndarray): The true values.
        - y_pred (numpy.ndarray): The predicted values.
        
        Returns:
        - mse (float): The mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def r_squared(self, y_true, y_pred):
        """
        Calculates the R-squared score between the true and predicted values.
        
        Parameters:
        - y_true (numpy.ndarray): The true values.
        - y_pred (numpy.ndarray): The predicted values.
        
        Returns:
        - r_squared (float): The R-squared score.
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    # Classification Metrics
    # TODO: Implement other classification metrics
    def accuracy(self, y_true, y_pred):
        """
        Calculates the accuracy score between the true and predicted values.
        
        Parameters:
        - y_true (numpy.ndarray): The true values.
        - y_pred (numpy.ndarray): The predicted values.
        
        Returns:
        - accuracy (float): The accuracy score.
        """
        return np.mean(y_true == y_pred)
    
    def precision(self, y_true, y_pred):
        """
        Calculates the precision score between the true and predicted values.
        
        Parameters:
        - y_true (numpy.ndarray): The true values.
        - y_pred (numpy.ndarray): The predicted values.
        
        Returns:
        - precision (float): The precision score.
        """
        # Ensure the arrays are numpy arrays for element-wise operations
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
               
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    
    def recall(self, y_true, y_pred):
        """
        Calculates the recall score between the true and predicted values.
        
        Parameters:
        - y_true (numpy.ndarray): The true values.
        - y_pred (numpy.ndarray): The predicted values.
        
        Returns:
        - recall (float): The recall score.
        """
        # Ensure the arrays are numpy arrays for element-wise operations
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)