import math

# --------------------------------------------------------
# DATASET – pure python
# --------------------------------------------------------
data = [
    [7.1, 8.0, 2.5, 350, 85],
    [6.8, 7.5, 3.0, 400, 80],
    [7.3, 8.2, 1.8, 320, 90],
    [6.5, 6.8, 4.0, 450, 70],
    [7.0, 7.8, 2.0, 360, 82],
    [6.7, 7.0, 3.8, 420, 75],
    [7.4, 8.3, 1.5, 300, 92],
    [6.6, 6.9, 4.2, 470, 68],
    [7.2, 8.1, 2.2, 340, 88],
    [6.9, 7.4, 3.1, 410, 78]
]

X = [row[:4] for row in data]
y = [row[4] for row in data]
N = len(X)

# --------------------------------------------------------
# Gaussian MF
# --------------------------------------------------------
def g(x, c, s):
    return math.exp(-0.5 * ((x - c) / s)**2)

# pH – 3 MFs (Low, Normal, High)
c_ph = [6.6, 7.0, 7.4]
s_ph = [0.25, 0.25, 0.25]

# DO – 3 MFs
c_do = [7.0, 7.6, 8.2]
s_do = [0.3, 0.3, 0.3]

# Turbidity – 2 MFs
c_tb = [2.0, 3.5]
s_tb = [0.8, 0.8]

# Conductivity – 2 MFs
c_cd = [330, 430]
s_cd = [50, 50]

# --------------------------------------------------------
# Manual 5×5 linear algebra
# --------------------------------------------------------
def mat_vec(A, v):
    r = [0]*5
    for i in range(5):
        for j in range(5):
            r[i] += A[i][j] * v[j]
    return r

def invert_5x5(M):
    n = 5
    A = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(M)]

    for i in range(n):
        pivot = A[i][i]
        if pivot == 0:
            for k in range(i+1, n):
                if A[k][i] != 0:
                    A[i], A[k] = A[k], A[i]
                    pivot = A[i][i]
                    break

        for j in range(2*n):
            A[i][j] /= pivot

        for k in range(n):
            if k != i:
                factor = A[k][i]
                for j in range(2*n):
                    A[k][j] -= factor * A[i][j]

    return [row[n:] for row in A]

# --------------------------------------------------------
# Forward Pass → 36 rules
# --------------------------------------------------------
def forward(x):
    ph, do, tb, cd = x

    mu_ph = [g(ph, c_ph[i], s_ph[i]) for i in range(3)]
    mu_do = [g(do, c_do[i], s_do[i]) for i in range(3)]
    mu_tb = [g(tb, c_tb[i], s_tb[i]) for i in range(2)]
    mu_cd = [g(cd, c_cd[i], s_cd[i]) for i in range(2)]

    w = []
    for a in range(3):
        for b in range(3):
            for c in range(2):
                for d in range(2):
                    w.append(mu_ph[a] * mu_do[b] * mu_tb[c] * mu_cd[d])

    s = sum(w)
    return [v/s for v in w] if s != 0 else w

# --------------------------------------------------------
# Train Consequents
# Each rule has: WQI = p1*ph + p2*do + p3*tb + p4*cd + b
# --------------------------------------------------------
def train_consequents(X, y):
    params = [[0]*5 for _ in range(36)]
    W_all = [forward(X[i]) for i in range(N)]

    for r in range(36):
        XtWX = [[0]*5 for _ in range(5)]
        XtWy = [0]*5

        for n in range(N):
            ph, do, tb, cd = X[n]
            row = [ph, do, tb, cd, 1]
            w = W_all[n][r]

            for i in range(5):
                for j in range(5):
                    XtWX[i][j] += w * row[i] * row[j]

            for i in range(5):
                XtWy[i] += w * row[i] * y[n]

        inv = invert_5x5(XtWX)
        params[r] = mat_vec(inv, XtWy)

    return params

# --------------------------------------------------------
# Predict
# --------------------------------------------------------
def predict(x, params):
    w = forward(x)
    outputs = []
    for r in range(36):
        p1, p2, p3, p4, b = params[r]
        ph, do, tb, cd = x
        outputs.append(p1*ph + p2*do + p3*tb + p4*cd + b)
    return sum(w[i] * outputs[i] for i in range(36))

# --------------------------------------------------------
# Train + Test
# --------------------------------------------------------
params = train_consequents(X, y)
preds = [predict(X[i], params) for i in range(N)]
rmse = math.sqrt(sum((preds[i] - y[i])**2 for i in range(N)) / N)

print("\n--- ANFIS WQI RESULTS (PURE PYTHON) ---\n")
print("Predictions:", [round(p,3) for p in preds])
print("Actual:", y)
print("RMSE:", rmse)
