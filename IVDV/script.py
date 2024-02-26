import numpy as np
from sklearn.linear_model import LinearRegression

# Independent variables (input data)
X = np.array([[1], [2], [3], [4], [5]])

# Dependent variables (output data)
y = np.array([2, 3, 4, 5, 6])

# Create and train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict dependent variables for new independent variables
new_X = np.array([[6], [7], [8]])
predictions = model.predict(new_X)

# Output
print("Independent variables (input data):")
print(X)

print("\nDependent variables (output data):")
print(y)

print("\nCoefficients (slope) of the trained linear regression model:", model.coef_[0])
print("Intercept of the trained linear regression model:", model.intercept_)

print("\nPredicted dependent variables for new independent variables:")
for i, x in enumerate(new_X):
    print("Independent variable:", x[0], "| Predicted dependent variable:", predictions[i])
