import numpy as np

# ---- Training Data ----
data = np.array([
    [0.1, 0.2, 0],   # class 0
    [0.2, 0.1, 0],
    [0.9, 1.0, 1],   # class 1
    [1.1, 0.9, 1]
])

# ---- Initialize codebook vectors ----
w0 = np.array([0.2, 0.1])    # class 0
w1 = np.array([1.0, 1.0])    # class 1
lr = 0.3

def closest(x):
    d0 = np.linalg.norm(x - w0)
    d1 = np.linalg.norm(x - w1)
    return 0 if d0 < d1 else 1

# ---- Training ----
for epoch in range(15):
    for x1, x2, label in data:
        x = np.array([x1, x2])
        c = closest(x)

        if c == label:   # move closer
            if c == 0: w0 += lr * (x - w0)
            else:       w1 += lr * (x - w1)
        else:            # move away
            if c == 0: w0 -= lr * (x - w0)
            else:       w1 -= lr * (x - w1)

# ---- Testing ----
for row in data:
    x, y, label = row
    print([x, y], "Classified as:", closest([x, y]))
