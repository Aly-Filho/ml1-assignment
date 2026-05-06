import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregar os dados reais
df = pd.read_csv('framingham.csv').dropna()
y = df['TenYearCHD'].values
X = df.drop('TenYearCHD', axis=1).values

df = pd.read_csv('framingham.csv')
df = df.dropna() #Remover linhas com valores nulos.

X = df.drop('TenYearCHD', axis=1).values
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)

X_train = (X - mu) / sigma
y_train = df['TenYearCHD'].values


m, n = X_train.shape


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(X, y, w, b):
    cost_sum = 0

    for i in range (m):
        z = np.dot(w, X[i]) + b
        g = sigmoid(z)

        cost_sum += -y[i] * np.log(g) - (1 - y[i]) * np.log(1 - g)

    return (1 / m) * cost_sum


def gradient_function(X, y, w, b):
    grad_w = np.zeros(n)
    grad_b = 0

    for i in range(m):
        z = np.dot(w, X[i]) + b
        g = sigmoid(z)

        grad_b += (g - y[i])

        for j in range (n):
            grad_w[j] += (g - y[i]) * X[i,j]

    grad_b = (1/m) * grad_b
    grad_w = (1/m) * grad_w

    return grad_b, grad_w


def gradient_descent(X, y, alpha, iterations):
    w = np.zeros(n)
    b = 0

    for i in range(iterations):
        grad_b, grad_w = gradient_function(X, y, w, b)

        w = w - alpha * grad_w
        b = b - alpha * grad_b

        if i % 1000 == 0:
            print(f"Iteration {i}: Cost {cost_function(X, y, w, b)}")
    
    return w, b


def predict(X, w, b):
    preds = np.zeros(m)

    for i in range(m):
        z = np.dot(w, X[i]) + b
        g = sigmoid(z)

        preds[i] = 1 if g >= 0.5 else 0
    
    return preds


learning_rate = 0.01
iterations = 10000

final_w, final_b = gradient_descent(X_train, y_train, learning_rate, iterations)

predictions = predict(X_train, final_w, final_b)
accuracy = np.mean(predictions == y_train) * 100
print(f"training accuracy: {accuracy:.2f}%")