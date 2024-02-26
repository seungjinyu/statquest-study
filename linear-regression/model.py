import numpy as np 
from sklearn.metrics import mean_squared_error, r2_score  # Importing mean_squared_error and r2_score functions from sklearn.metrics module

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.num_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X , self.weights) + self.bias

# Sample data
X_train = np.array([[1, 1.5], [2, 2.5], [3, 3.5], [4, 4.5], [5, 5.5]])  # Training input data (features)
y_train = np.array([2, 3, 4, 5, 6])  # Training output data (target)
X_test = np.array([[6, 6.5], [7, 7.5], [8, 8.5]])  # Test input data (features)
y_test = np.array([7, 8, 9])  # Test output data (target)

# Create and train the linear regression model
model = LinearRegression()  # Create a LinearRegression model object
model.fit(X_train, y_train)  # Train the model using the training data

# Make predictions using the trained model
predictions = model.predict(X_test)  # Make predictions on the test data

# Print predictions
for i in range(len(predictions)):
    print("Prediction:", predictions[i])  # Print each prediction made by the model

# Test the predictions by comparing them with the actual values
for i in range(len(predictions)):
    print("Predicted:", predictions[i], "Actual:", y_test[i])  # Compare each prediction with the corresponding actual value

# Calculate evaluation metrics (Mean Squared Error and R-squared)
mse = mean_squared_error(y_test, predictions)  # Calculate the Mean Squared Error
r2 = r2_score(y_test, predictions)  # Calculate the R-squared
print("Mean Squared Error:", mse)  # Print the Mean Squared Error
print("R-squared:", r2)  # Print the R-squared