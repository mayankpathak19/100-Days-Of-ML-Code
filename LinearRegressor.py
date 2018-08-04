import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


# Generating the data and plotting it
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)
plt.plot(x, y, 'b.')
plt.xlabel('x1')
plt.ylabel('y')
plt.axis([0, 2, 0, 15])
plt.show()
x_new = np.array([[0], [2]])
x_b = np.c_[np.ones((100, 1)), x]

# Without using sklearn
# Solving the normal equation
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
print(theta_best)

# Making Predictions using theta_best
x_new_b = np.c_[np.ones((2, 1)), x_new]
y_predict = x_new_b.dot(theta_best)
print(y_predict)

# Plotting the model's prediction
plt.plot(x_new, y_predict, 'r-')
plt.plot(x, y, 'b.')
plt.xlabel('x1')
plt.ylabel('y')
plt.axis([0, 2, 0, 15])
plt.legend(['Prediction', 'Data'], loc='upper left')
plt.show()


# Using sklearn
linear_model = LinearRegression()
linear_model.fit(x, y)
print(linear_model.intercept_, linear_model.coef_)
y_predict = linear_model.predict(x_new)
print(y_predict)
