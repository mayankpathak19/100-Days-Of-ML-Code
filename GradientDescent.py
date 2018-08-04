import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor


# Generating the data and plotting it
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)
m = 100
x_b = np.c_[np.ones((100, 1)), x]
plt.plot(x, y, 'b.')
plt.xlabel('x1')
plt.ylabel('y')
plt.axis([0, 2, 0, 15])

"""
Batch Gradient Descent without sklearn
"""


x_new = np.array([[0], [2]])


# Learning rate
eta = 0.5
n_iterations = 1000

# Random initialization
theta = np.random.randn(2, 1)


for iterations in range(n_iterations):
    gradients = (2 / m) * x_b.T.dot(x_b.dot(theta) - y)
    theta = theta - eta * gradients
    plt.plot(x_b, theta[1][0] + theta[0][0] * x_b)
plt.xlabel('x')
plt.ylabel('y')
plt.axis([0, 2, 0, 15])
plt.show()


"""
Stochastic Gradient Descent without sklearn
"""
n_epochs = 50

# Learning Schedule Hyper-Parameters
t0, t1 = 5, 50

# Random Initialization
theta = np.random.randn(2, 1)


def learning_schedule(t):
    return t0 / (t + t1)


for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = x_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta -= (eta * gradients)
plt.plot(x_b, theta[1][0] + theta[0][0] * x_b)
plt.xlabel('x')
plt.ylabel('y')
plt.axis([0, 2, 0, 15])
plt.show()


"""
Stochastic Gradient Descent with sklearn
"""


model = SGDRegressor(n_iter=50, penalty=None, eta0=0.01)
model.fit(x, y.ravel())
print(model.intercept_, model.coef_)