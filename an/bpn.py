import math
import random

# --------------------------
# Activation functions
# --------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# --------------------------
# Training data (XOR)
# --------------------------

data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

# --------------------------
# Initialize weights
# --------------------------

w1 = random.random()
w2 = random.random()
w3 = random.random()
w4 = random.random()

w5 = random.random()
w6 = random.random()

lr = 0.5   # learning rate

# --------------------------
# Training Loop
# --------------------------

for epoch in range(5000):
    for x, target in data:
        x1, x2 = x

        # ---- Forward Pass ----
        h1 = sigmoid(x1*w1 + x2*w2)
        h2 = sigmoid(x1*w3 + x2*w4)

        output = sigmoid(h1*w5 + h2*w6)

        # ---- Error ----
        error = target - output

        # ---- Backpropagation ----
        d_output = error * sigmoid_derivative(output)

        d_w5 = d_output * h1
        d_w6 = d_output * h2

        # Hidden layer error
        d_h1 = d_output * w5 * sigmoid_derivative(h1)
        d_h2 = d_output * w6 * sigmoid_derivative(h2)

        d_w1 = d_h1 * x1
        d_w2 = d_h1 * x2
        d_w3 = d_h2 * x1
        d_w4 = d_h2 * x2

        # ---- Weight Update ----
        w1 += lr * d_w1
        w2 += lr * d_w2
        w3 += lr * d_w3
        w4 += lr * d_w4
        w5 += lr * d_w5
        w6 += lr * d_w6

# --------------------------
# Testing ANN After Training
# --------------------------

print("Testing the trained network:")
for x, t in data:
    x1, x2 = x
    h1 = sigmoid(x1*w1 + x2*w2)
    h2 = sigmoid(x1*w3 + x2*w4)
    out = sigmoid(h1*w5 + h2*w6)
    print(f"Input: {x}  Output: {out:.4f}")
