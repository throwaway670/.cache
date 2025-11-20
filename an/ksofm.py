import numpy as np

# ---- Input Data ----
data = np.array([[0.2, 0.1],
                 [0.9, 1.0],
                 [0.3, 0.2],
                 [1.1, 0.9]])

# ---- SOM weights (2 neurons) ----
W = np.random.rand(2, 2)
lr = 0.3

def winner(x):
    d = np.linalg.norm(W - x, axis=1)
    return np.argmin(d)

# ---- Training ----
for epoch in range(20):
    for x in data:
        i = winner(x)
        W[i] += lr * (x - W[i])

# ---- Test ----
for x in data:
    print("Input:", x, "Winner neuron:", winner(x))
