
class AutoRegressor:
    pass


# This will be a class that takes in a dataset and automatically selects the best regression model for the data.
# It will use a variety of regression models and compare their performance using metrics such as R-squared, RMSE, etc.
# It will also provide a summary of the models used and their performance.
# The class will have methods to fit the models, predict, and evaluate the performance of the models.

# Predicted Usage:
# ------------------------------------------------------ 
# from sega_learn.auto import AutoRegressor
# from sklearn import datasets
# from sklearn.utils import shuffle
# import numpy as np
# boston = datasets.load_boston()
# X, y = shuffle(boston.data, boston.target, random_state=13)
# X = X.astype(np.float32)
# offset = int(X.shape[0] * 0.9)
# X_train, y_train = X[:offset], y[:offset]
# X_test, y_test = X[offset:], y[offset:]
# reg = AutoRegressor()
# models,predictions = reg.fit(X_train, X_test, y_train, y_test)
# models

# | Model                         |   R-Squared |     RMSE |   Time Taken |
# |:------------------------------|------------:|---------:|-------------:|
# | SVR                           |   0.877199  |  2.62054 |    0.0330021 |
# | RandomForestRegressor         |   0.874429  |  2.64993 |    0.0659981 |
# | AdaBoostRegressor             |   0.865851  |  2.73895 |    0.144999  |
# ...
# So on..