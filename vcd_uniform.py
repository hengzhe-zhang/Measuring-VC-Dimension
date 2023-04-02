import math

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

vcds = []
# Train a linear regression estimator on the dataset
model = LinearRegression()

for n_samples in [10, 20, 30, 50, 100, 200, 500]:
    deltas = []
    # Number of trails
    m = 50
    for _ in range(0, m):
        # Generate a random dataset
        X = np.random.random(size=(n_samples, 5))
        y_class = np.random.randint(0, 1, size=(n_samples,))

        # Normal
        Z1 = np.hstack((X, y_class.reshape(-1, 1)))
        # Reverse
        Z2 = np.hstack((X, (1 - y_class).reshape(-1, 1)))
        n1 = int(len(X) / 2)
        Z1_1 = Z1[:n1]
        Z1_2 = Z1[n1:]
        Z2_2 = Z2[n1:]
        model.fit(np.vstack((Z1_1[:, :-1], Z2_2[:, :-1])),
                  np.vstack((Z1_1[:, -1].reshape(-1, 1), Z2_2[:, -1].reshape(-1, 1))))
        E1 = mean_absolute_error(Z1_1[:, -1], model.predict(Z1_1[:, :-1]) > 0.5)
        E2 = mean_absolute_error(Z1_2[:, -1], model.predict(Z1_2[:, :-1]) > 0.5)
        delta = abs(E1 - E2)
        deltas.append(delta)
    deltas = np.mean(deltas)


    def phi(tau):
        # Estimate the effective VC-dimension by observing the maximum deviation delta of error rates
        a = 0.16
        b = 1.2
        k = 0.14928

        if tau < 0.5:
            return 1
        else:
            numerator = a * (math.log(2 * tau) + 1)
            denominator = tau - k
            temp = b * (tau - k) / (math.log(2 * tau) + 1)
            radicand = 1 + temp
            return numerator / denominator * (math.sqrt(radicand) + 1)


    # calculate VC dimension
    h = np.arange(1, 100)
    en = (n_samples / 2) / h
    eps = (np.array(list(map(phi, en))) - deltas) ** 2
    h_est = h[np.argmin(eps)]
    print("Estimated VC-dimension:", h_est)
