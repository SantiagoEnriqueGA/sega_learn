
class AutoClassifier:
    pass


# This will be a class that takes in a dataset and automatically selects the best classification model for the data.
# It will use a variety of classification models and compare their performance using metrics such as accuracy, precision, recall, etc.
# It will also provide a summary of the models used and their performance.
# The class will have methods to fit the models, predict, and evaluate the performance of the models.

# Predicted Usage:
# ------------------------------------------------------
# from sega_learn.auto import AutoClassifier
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# data = load_breast_cancer()
# X = data.data
# y= data.target
# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
# clf = AutoClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
# models,predictions = clf.fit(X_train, X_test, y_train, y_test)
# models


# | Model                          |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |   Time Taken |
# |:-------------------------------|-----------:|--------------------:|----------:|-----------:|-------------:|
# | LinearSVC                      |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0150008 |
# | SGDClassifier                  |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0109992 |
# | LogisticRegression             |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.0200036 |
# ...
# So on..