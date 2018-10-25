import numpy as np
import math

def predict(poly_para_K, poly_x, theta):
    poly_x = poly_x.reshape(poly_x.shape[0], 1)
    X = poly_x ** 0
    for i in range(1, poly_para_K+1):
        temp = poly_x ** i
        X = np.hstack((X, temp))
    poly_x = X
    predict_y = np.dot(poly_x, theta)  #poly_x is a 1*K
    return predict_y

def gaussian( mu, var, x):
    y = 1.0*(np.exp(-(x - mu) ** 2 / (2.0 * var)) / (((2.0 * np.pi * var) ** 0.5)))
    # print(y)
    # y = math.exp(-1.0*(x-mu)**2/(2.0*var)) / (math.sqrt(2*math.pi*var))
    return y

def pref_BR(mu_estimator, var_estimator,poly_para_K, poly_x):
    x = poly_x
    poly_x = poly_x.reshape(poly_x.shape[0], 1)
    X = poly_x ** 0
    for i in range(1, poly_para_K + 1):
        temp = poly_x ** i
        X = np.hstack((X, temp))
    poly_x = X
    # print(poly_x.shape)
    predict_y = []
    mu = []
    for i in range(len(poly_x)):
        mu_star = np.dot(poly_x[i], mu_estimator)
        mu.append(mu_star)
        # print(mu_star)
        # temp = poly_x[i].reshape(poly_x[i].shape[0],1)
        # print(poly_x[i].shape)
        var_star = np.dot(np.dot(poly_x[i], var_estimator), poly_x[i].T)
        # var_star = np.dot(np.dot(temp.T, var_estimator),temp)
        # print(var_star)
        y = gaussian(mu_star, var_star,(poly_x[i][1]))
        # y = gaussian(mu_estimator, var_star, mu_star)
        # print(poly_x[i][1])
        # print(y)
        predict_y.append(y)
    # print(np.ravel(np.array(predict_y)))
    return np.ravel(np.array(predict_y)) , np.ravel(np.array(mu))