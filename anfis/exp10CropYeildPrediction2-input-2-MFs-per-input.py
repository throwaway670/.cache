# ---------------------------------------------------------
# ANFIS for Crop Yield Prediction — PURE PYTHON (NO NUMPY)
# ---------------------------------------------------------
import math

# ---------------------------------------------------------
# DATASET (Rainfall, Nitrogen, Yield)
# ---------------------------------------------------------
data = [
    [520, 1.1, 2.8],
    [610, 1.3, 3.2],
    [450, 0.9, 2.4],
    [700, 1.5, 3.8],
    [480, 1.0, 2.5],
    [750, 1.6, 4.0],
    [550, 1.2, 3.0],
    [430, 0.8, 2.2],
    [680, 1.4, 3.6],
    [510, 1.1, 2.9],
    [720, 1.5, 3.9],
]

X = [row[:2] for row in data]   # Rainfall, Nitrogen
y = [row[2] for row in data]
N = len(X)


# ---------------------------------------------------------
# Gaussian Membership Function
# ---------------------------------------------------------
def gaussian(x, c, s):
    return math.exp(-0.5 * ((x - c) / s) ** 2)


# MF parameters
c_R = [480, 680]      # Rainfall centers: Low, High
s_R = [80, 80]

c_N = [1.0, 1.4]      # Nitrogen centers: Low, High
s_N = [0.15, 0.15]


# ---------------------------------------------------------
# FORWARD PASS
# ---------------------------------------------------------
def forward(x):
    Rain, Nitro = x

    # Layer 1 — Membership values
    mu_R = [
        gaussian(Rain, c_R[0], s_R[0]),
        gaussian(Rain, c_R[1], s_R[1])
    ]
    mu_N = [
        gaussian(Nitro, c_N[0], s_N[0]),
        gaussian(Nitro, c_N[1], s_N[1])
    ]

    # Layer 2 — Rule firing strengths
    w = [
        mu_R[0] * mu_N[0],   # Low Rain, Low N
        mu_R[0] * mu_N[1],   # Low Rain, High N
        mu_R[1] * mu_N[0],   # High Rain, Low N
        mu_R[1] * mu_N[1],   # High Rain, High N
    ]

    # Layer 3 — Normalized weights
    wsum = sum(w)
    if wsum == 0:
        w_norm = w
    else:
        w_norm = [wi / wsum for wi in w]

    return w_norm


# ---------------------------------------------------------
# HELPER: 3x3 matrix inverse (Gauss-Jordan)
# ---------------------------------------------------------
def invert_3x3(M):
    size = 3
    # Augment matrix with identity
    aug = [M[i] + [1 if i == j else 0 for j in range(size)] for i in range(size)]

    # Gauss-Jordan elimination
    for col in range(size):
        pivot = aug[col][col]
        if pivot == 0:
            raise ValueError("Matrix not invertible")

        # Normalize pivot row
        for j in range(col, 2 * size):
            aug[col][j] /= pivot

        # Eliminate other rows
        for i in range(size):
            if i != col:
                factor = aug[i][col]
                for j in range(col, 2 * size):
                    aug[i][j] -= factor * aug[col][j]

    # Extract inverse
    inv = [row[size:] for row in aug]
    return inv


# ---------------------------------------------------------
# TRAIN CONSEQUENT PARAMETERS using Weighted Least Squares
# ---------------------------------------------------------
def train_consequents():
    X_aug = [[x[0], x[1], 1.0] for x in X]   # [Rain, Nitrogen, 1]
    W_all = [forward(X[i]) for i in range(N)]

    params = [[0,0,0] for _ in range(4)]    # 4 rules → (p, q, r)

    for r in range(4):
        # Build XtWX (3x3) and XtWy manually
        XtWX = [[0]*3 for _ in range(3)]
        XtWy = [0]*3

        for i in range(N):
            w = W_all[i][r]
            Xi = X_aug[i]

            for a in range(3):
                for b in range(3):
                    XtWX[a][b] += w * Xi[a] * Xi[b]

            for a in range(3):
                XtWy[a] += w * Xi[a] * y[i]

        # Solve for parameters
        inv = invert_3x3(XtWX)
        solution = [sum(inv[row][j] * XtWy[j] for j in range(3)) for row in range(3)]
        params[r] = solution

    return params


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
def predict(x, params):
    w_norm = forward(x)
    outputs = []

    for r in range(4):
        p, q, r0 = params[r]
        outputs.append(p*x[0] + q*x[1] + r0)

    return sum(outputs[i] * w_norm[i] for i in range(4))


# ---------------------------------------------------------
# TRAIN + TEST
# ---------------------------------------------------------
params = train_consequents()
preds = [predict(X[i], params) for i in range(N)]

rmse = math.sqrt(sum((preds[i] - y[i])**2 for i in range(N)) / N)


# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------
print("\n--- ANFIS CROP YIELD (PURE PYTHON) ---")

print("\nRule Consequent Parameters (p, q, r):")
for p in params:
    print(p)

print("\nPredictions:", [round(v, 3) for v in preds])
print("Actual     :", y)
print("\nRMSE =", rmse)
