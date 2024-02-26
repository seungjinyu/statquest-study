import numpy as np
from sklearn.model_selection import KFold, cross_val_score, LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# example data set
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
y = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# create linear regression model 
model = LinearRegression()

# 3 fold cross validation
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold)
print("3 fold cross validation mean score:", np.mean(scores))

# 5 fold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold)
print("5 fold cross validation mean score:", np.mean(scores))
