import numpy as np

# ---- Training Patterns ----
patterns = np.array([
    [1, -1, 1, -1],
    [1, 1, -1, -1]
])

# ---- Weight Matrix ----
n = patterns.shape[1]
W = np.zeros((n, n))

for p in patterns:
    W += np.outer(p, p)

np.fill_diagonal(W, 0)

# ---- Update Function ----
def hopfield_recall(state):
    for i in range(len(state)):
        net = np.dot(W[i], state)
        state[i] = 1 if net >= 0 else -1
    return state

# ---- Test ----
test = np.array([1, -1, -1, -1])   # Noisy input
output = hopfield_recall(test.copy())

print("Input :", test)
print("Output:", output)
