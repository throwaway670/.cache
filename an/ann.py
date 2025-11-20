import math

# Activation functions
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return math.tanh(x)

# ANN class
class SimpleANN:
    def __init__(self):
        # Hardcoded weights for demonstration (you can randomize)
        self.w1 = 0.5
        self.w2 = -1.2
        self.w3 = 1.0
        self.w4 = 0.8
        self.w_out1 = 1.1
        self.w_out2 = -0.6

    def forward(self, x):
        # Hidden layer (ReLU)
        h1 = relu(x * self.w1)
        h2 = relu(x * self.w2)

        # Output layer (Sigmoid)
        out = sigmoid(h1 * self.w_out1 + h2 * self.w_out2)
        return out

# Testing ANN
ann = SimpleANN()
inputs = [-3, -1, 0.5, 2, 4, -6]

for x in inputs:
    y = ann.forward(x)
    print(f"Input: {x:>3}, Output: {y:.4f}")
