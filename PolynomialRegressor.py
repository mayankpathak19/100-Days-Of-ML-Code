import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# Generating the data and plotting it
m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * (x ** 2) + x + 2 + np.random.randn(m, 1)
plt.plot(x, y, 'b.')
plt.xlabel('x1')
plt.ylabel('y')
plt.axis([-3, 3, 0, 10])
# plt.show()


polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
x_polynomial = polynomial_features.fit_transform(x)
print(x[0])
print(x_polynomial[0])
model = LinearRegression()
model.fit(x_polynomial, y)
print(model.intercept_, model.coef_)
