import numpy as np

# Activation Function (Binary Step)
def activation(x):
    return 1 if x >= 0 else 0

# Single Layer Perceptron Training
def train_perceptron(X, Y, lr=0.1, epochs=20):
    w = np.zeros(X.shape[1])
    b = 0

    for epoch in range(epochs):
        print("\nEpoch", epoch+1)
        for i in range(len(X)):
            net = np.dot(X[i], w) + b
            y_pred = activation(net)
            error = Y[i] - y_pred

            # Weight Update Rule
            w = w + lr * error * X[i]
            b = b + lr * error

            print(f"Input:{X[i]}, Pred:{y_pred}, Error:{error}, Weights:{w}, Bias:{b}")

    return w, b

# Prediction Function
def predict(X, w, b):
    for x in X:
        net = np.dot(x, w) + b
        print(x, " -> ", activation(net))