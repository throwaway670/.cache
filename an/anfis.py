import numpy as np

# ---------------------------
# 1. Membership functions
# ---------------------------
def low(x):
    return max(0, min((5 - x) / 5, 1))  # linear decreasing

def high(x):
    return max(0, min((x - 5) / 5, 1))  # linear increasing


# ---------------------------
# 2. Dataset (Example)
# ---------------------------
# X1 = Rainfall, X2 = Nitrogen
X = np.array([
    [2, 3],
    [4, 5],
    [6, 4],
    [8, 7]
], dtype=float)

# Target output (Crop yield)
y = np.array([10, 15, 18, 25], dtype=float)


# ---------------------------
# 3. Initialize rule parameters
# Output for each rule:
# f = p*x1 + q*x2 + r
# 4 rules → 4 sets of parameters
# ---------------------------
params = np.random.rand(4, 3)   # (p, q, r) for each rule


# ---------------------------
# 4. Train ANFIS (Simple Least Squares)
# ---------------------------
lr = 0.01   # learning rate

for epoch in range(200):
    pred_all = []

    for i, (x1, x2) in enumerate(X):

        # ---------------------------
        # Membership values
        # ---------------------------
        x1_low = low(x1)
        x1_high = high(x1)
        x2_low = low(x2)
        x2_high = high(x2)

        # ---------------------------
        # 4 Rules:
        # R1: Low, Low
        # R2: Low, High
        # R3: High, Low
        # R4: High, High
        # ---------------------------
        w1 = x1_low  * x2_low
        w2 = x1_low  * x2_high
        w3 = x1_high * x2_low
        w4 = x1_high * x2_high

        W = np.array([w1, w2, w3, w4])

        # Rule outputs (Sugeno)
        o = np.zeros(4)
        for r in range(4):
            p, q, r0 = params[r]
            o[r] = p * x1 + q * x2 + r0

        # Weighted output
        y_pred = np.sum(W * o) / np.sum(W)
        pred_all.append(y_pred)

        # ---------------------------
        # Update parameters (simple gradient)
        # ---------------------------
        error = y[i] - y_pred
        params += lr * error  # simple update


    if epoch % 50 == 0:
        print("Epoch", epoch, "Error =", np.mean((y - np.array(pred_all))**2))


# ---------------------------
# 5. Final Predictions
# ---------------------------
print("\nFinal Predictions:")
print(pred_all)


