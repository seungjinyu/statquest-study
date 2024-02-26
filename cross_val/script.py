import numpy as np
from sklearn.model_selection import KFold, cross_val_score, LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd


# 예시 데이터셋 생성
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
y = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# 선형 회귀 모델 생성
model = LinearRegression()

# 3 폴드 교차 검증
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold)
print("3 폴드 교차 검증 평균 점수:", np.mean(scores))

# 5 폴드 교차 검증
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold)
print("5 폴드 교차 검증 평균 점수:", np.mean(scores))

# Data preparation
data = {
    "area": [100, 120, 150, 180, 200],
    "number_of_rooms": [2, 3, 4, 5, 5],
    "price": [500000, 600000, 800000, 1000000, 1200000]
}

# Validate data integrity
assert len(set(len(value) for value in data.values())) == 1, "Data has inconsistent lengths"
for key, value in data.items():
    assert not any(pd.isna(val) for val in value), f"Missing values found in '{key}'"

# LOO cross-validation and average error calculation
model = LinearRegression()
error_list = []

for i in range(len(data["area"])):
    # Create training and test data subsets with consistent lengths
    train_data_X = {key: value for key, value in data.items() if key != "price"}  # Features
    train_data_y = data["price"].copy()  # Target variable

    # Correctly remove corresponding element for test data
    del train_data_X[list(train_data_X.keys())[i]]  # Remove feature
    del train_data_y[i]  # Remove corresponding target

    # Check for valid data lengths
    assert len(list(train_data_X.values())) == len(train_data_y), "Inconsistent data lengths"

    # Train the model
    model.fit(list(train_data_X.values()), train_data_y)

    # Prediction and error calculation
    test_data_X = {key: value[i:i+1] for key, value in data.items() if key != "price"}
    prediction = model.predict(list(test_data_X.values()))
    error = mean_squared_error(data["price"][i:i+1], prediction)
    error_list.append(error)

average_error = sum(error_list) / len(error_list)

# Print results
print("Average error:", average_error)