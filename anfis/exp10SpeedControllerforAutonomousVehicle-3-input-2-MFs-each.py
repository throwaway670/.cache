# --------------------------------------------------------
# PURE PYTHON ANFIS – NO NUMPY / NO LIBRARIES
# --------------------------------------------------------

import math

# ---------------------------
# B2 SPEED CONTROLLER DATASET
# ---------------------------
# Inputs: Distance, Slope, Speed
# Output: Acceleration (+) or Brake (-)
data = [
    [50, 0, 40, 5],
    [20, -2, 45, -8],
    [10, 1, 30, -10],
    [80, 3, 50, 4],
    [15, 0, 35, -9],
    [100, -1, 60, 6],
    [25, 2, 42, -5],
    [70, -3, 55, 5],
    [12, 1, 38, -10],
    [90, 0, 58, 3]
]

X = [row[:3] for row in data]
y = [row[3] for row in data]
N = len(X)


# --------------------------------------------------------
# Gaussian Membership Function
# --------------------------------------------------------
def gauss(x, c, s):
    return math.exp(-0.5 * ((x - c) / s) ** 2)


# --------------------------------------------------------
# Membership Function Parameters (2 per input)
# --------------------------------------------------------
# Distance MFs
c_D = [20, 80]           # centers
s_D = [15, 15]           # sigmas

# Slope MFs
c_S = [-2, 2]
s_S = [1.5, 1.5]

# Speed MFs
c_V = [35, 55]
s_V = [10, 10]


# --------------------------------------------------------
# Forward Pass – Compute 8 Rule Weights
# --------------------------------------------------------
def forward(x):
    d, s, v = x

    # Degrees of membership
    mu_d = [gauss(d, c_D[i], s_D[i]) for i in range(2)]
    mu_s = [gauss(s, c_S[i], s_S[i]) for i in range(2)]
    mu_v = [gauss(v, c_V[i], s_V[i]) for i in range(2)]

    # 2 × 2 × 2 = 8 rule firing strengths
    w = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                w.append(mu_d[i] * mu_s[j] * mu_v[k])

    # Normalize
    ssum = sum(w)
    if ssum == 0:
        return w
    return [wi / ssum for wi in w]


# --------------------------------------------------------
# Manual 4x4 Matrix Inverse (Gauss-Jordan)
# --------------------------------------------------------
def invert_4x4(M):
    size = 4
    aug = [M[i] + [1 if i == j else 0 for j in range(size)] for i in range(size)]

    for col in range(size):
        pivot = aug[col][col]
        if pivot == 0:
            raise ValueError("Matrix not invertible")
        for j in range(col, 2 * size):
            aug[col][j] /= pivot
        for i in range(size):
            if i != col:
                factor = aug[i][col]
                for j in range(col, 2 * size):
                    aug[i][j] -= factor * aug[col][j]

    return [row[size:] for row in aug]


# --------------------------------------------------------
# Train Consequents (Weighted Least Squares)
# --------------------------------------------------------
def train_consequents():
    X_aug = [x + [1] for x in X]     # add bias
    W_all = [forward(X[i]) for i in range(N)]

    params = [[0, 0, 0, 0] for _ in range(8)]    # 8 rules × 4 parameters

    for r in range(8):
        XtWX = [[0]*4 for _ in range(4)]
        XtWy = [0]*4

        for i in range(N):
            w = W_all[i][r]
            Xi = X_aug[i]

            # XtWX += w * Xi * Xi^T
            for a in range(4):
                for b in range(4):
                    XtWX[a][b] += w * Xi[a] * Xi[b]

            # XtWy += w * Xi * y
            for a in range(4):
                XtWy[a] += w * Xi[a] * y[i]

        invM = invert_4x4(XtWX)
        params[r] = [sum(invM[row][j] * XtWy[j] for j in range(4)) for row in range(4)]

    return params


# --------------------------------------------------------
# Prediction
# --------------------------------------------------------
def predict(x, params):
    w = forward(x)
    outputs = []

    for r in range(8):
        a, b, c, d = params[r]
        f = a*x[0] + b*x[1] + c*x[2] + d
        outputs.append(f)

    return sum(outputs[i] * w[i] for i in range(8))


# --------------------------------------------------------
# TRAIN + TEST
# --------------------------------------------------------
params = train_consequents()
preds = [predict(X[i], params) for i in range(N)]

rmse = math.sqrt(sum((preds[i] - y[i])**2 for i in range(N)) / N)


# --------------------------------------------------------
# RESULTS
# --------------------------------------------------------
print("\n--- ANFIS SPEED CONTROLLER (PURE PYTHON, NO LIBRARIES) ---")
print("\nRule Parameters (a, b, c, d):")
for p in params:
    print(p)

print("\nPredictions :", [round(p, 2) for p in preds])
print("Actual      :", y)
print("\nRMSE =", rmse)
