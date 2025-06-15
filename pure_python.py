#�� Step 1: Load data manually
# Sample dataset
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
# �� Step 2: Initialize weights and other terms
# Initial values
w1 = 0  # for Hours
w2 = 0  # for Sleep
b = 0   # bias

lr = 0.01
epochs = 1000

#�� Step 3: Define cost function (MSE) and predict y on every new iteration
def predict(x1, x2, w1, w2, b):
    return w1 * x1 + w2 * x2 + b

def compute_cost(data, w1, w2, b):
    total_error = 0
    n = len(data)
    for x1, x2, y in data:
        y_pred = predict(x1, x2, w1, w2, b)
        total_error += (y - y_pred) ** 2
    return total_error / n

#�� Step 4: Gradient Descent loop

cost_history = []

for epoch in range(epochs):
    dw1 = 0
    dw2 = 0
    db = 0
    n = len(data)

    for x1, x2, y in data:
        y_pred = predict(x1, x2, w1, w2, b)
        error = y - y_pred
        dw1 += -2 * x1 * error
        dw2 += -2 * x2 * error
        db += -2 * error

    # Average gradients
    dw1 /= n
    dw2 /= n
    db /= n

    # Update weights
    w1 -= lr * dw1
    w2 -= lr * dw2
    b -= lr * db

    cost = compute_cost(data, w1, w2, b)
    cost_history.append(cost)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Cost: {cost:.4f}")
#�� Step 5: Final weights
print(f"w1: {w1}, w2: {w2}, b: {b}")
import matplotlib.pyplot as plt

#�� Cost vs Iteration plot
plt.plot(range(epochs),cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations (Pure Python)")
plt.grid(True)
plt.legend
plt.show()

import math

def evaluate_metrics(data, w1, w2, b):
    n = len(data)
    mae = 0
    mse = 0
    y_true = []
    y_pred = []

    for x1, x2, y in data:
        yp = predict(x1, x2, w1, w2, b)
        y_true.append(y)
        y_pred.append(yp)
        mae += abs(y - yp)
        mse += (y - yp) ** 2

    mae /= n
    rmse = math.sqrt(mse / n)

    # R² Score
    y_mean = sum(y_true) / n
    ss_total = sum((y - y_mean) ** 2 for y in y_true)
    ss_res = sum((y - yp) ** 2 for y, yp in zip(y_true, y_pred))
    r2 = 1 - (ss_res / ss_total)

    return mae, rmse, r2
    
mae, rmse, r2_score = evaluate_metrics(data, w1, w2, b)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2_score:.4f}")
