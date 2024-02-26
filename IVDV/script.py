import numpy as np
from sklearn.linear_model import LinearRegression

# 독립 변수 (입력 데이터)
X = np.array([[1], [2], [3], [4], [5]])

# 종속 변수 (출력 데이터)
y = np.array([2, 3, 4, 5, 6])

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 새로운 독립 변수 값에 대한 종속 변수 예측
new_X = np.array([[6], [7], [8]])
predictions = model.predict(new_X)

# 결과 출력
print("독립 변수 (입력 데이터):")
print(X)

print("\n종속 변수 (출력 데이터):")
print(y)

print("\n학습된 선형 회귀 모델의 계수 (기울기):", model.coef_[0])
print("학습된 선형 회귀 모델의 절편:", model.intercept_)

print("\n새로운 독립 변수 값에 대한 종속 변수 예측:")
for i, x in enumerate(new_X):
    print("독립 변수:", x[0], "| 예측된 종속 변수:", predictions[i])
