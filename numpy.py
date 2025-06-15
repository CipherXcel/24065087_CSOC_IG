import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


data = [
    [1, 8, 20],
    [2, 8, 25],
    [2.5, 7.5, 28],
    [3, 7, 32],
    [3.5, 7, 36],
    [4, 6.5, 40],
    [4.5, 6, 43],
    [5, 6, 48],
    [5.5, 5.5, 52],
    [6, 5, 56],
    [6.5, 5, 60],
    [7, 4.5, 64],
    [7.5, 4, 67],
    [8, 4, 70],
    [8.5, 3.5, 73],
    [9, 3, 76],
    [9.5, 3, 79],
    [10, 2.5, 82]
]

data=np.array(data)
model=LinearRegression()
x=data[:,:2]
y=data[:,2].reshape(-1,1)
model.fit(x,y)
predictions=model.predict(x)

print("w",model.coef_)
print ("b",model.intercept_)

print("MAE:", np.mean(np.abs(y-predictions)))
print("RMSE:", np.sqrt(np.mean((y-predictions)**2)))
print("R2_score:", r2_score(y,predictions))

import matplotlib.pyplot as plt

plt.scatter(y, predictions, color='green', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Fit Line')
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted (Scikit-learn)")
plt.legend()
plt.grid(True)
plt.show()
