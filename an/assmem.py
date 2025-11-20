import numpy as np

# -----------------------
# Training patterns
# -----------------------
X = np.array([
    [1, -1, 1, -1],    # Input pattern 1
    [-1, 1, -1, 1]     # Input pattern 2
])

Y = np.array([
    [1, 1],            # Output pattern 1
    [-1, -1]           # Output pattern 2
])

# -----------------------
# Hebbian Learning Rule
# W = Σ (X_i^T * Y_i)
# -----------------------
W = np.zeros((4, 2))

for i in range(len(X)):
    W += np.outer(X[i], Y[i])   # weight update

print("Weight Matrix:\n", W)

# -----------------------
# Recall Phase
# Y' = sign(X_test * W)
# -----------------------
def recall(x):
    y = np.dot(x, W)
    return np.where(y >= 0, 1, -1)

# -----------------------
# Test with noisy input
# -----------------------
test = np.array([1, -1, -1, -1])   # noisy version of X1

output = recall(test)
print("\nInput :", test)
print("Output:", output)
