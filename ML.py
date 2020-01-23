### The ML based evaluation
import random
import numpy as np
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold

def get_score(X, y, model):

    folds = KFold(n_splits=3, shuffle=True, random_state=1)
    scorer = make_scorer(r2_score)

    scores = []
    for train_index, test_index in folds.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        scores.append(scorer(model, X_test, y_test))

    return np.mean(scores)