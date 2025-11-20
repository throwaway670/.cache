import math

# --------------------------------------------------------
# DATASET (pure python lists)
# --------------------------------------------------------
data = [
    [150, 0.15, 10, 45, 0.12],
    [180, 0.20, 12, 55, 0.18],
    [200, 0.25, 15, 60, 0.25],
    [160, 0.18, 11, 48, 0.15],
    [170, 0.22, 14, 58, 0.20],
    [210, 0.28, 18, 65, 0.30],
    [155, 0.16, 10, 46, 0.13],
    [190, 0.24, 16, 62, 0.23],
    [175, 0.21, 13, 54, 0.18],
    [220, 0.30, 20, 70, 0.35]
]

X = [row[:4] for row in data]
y = [row[4] for row in data]
N = len(X)

# --------------------------------------------------------
# GAUSSIAN MF
# --------------------------------------------------------
def g(x, c, s):
    return math.exp(-0.5 * ((x - c) / s) ** 2)

# Cutting speed – 3 MFs
c_cs = [160, 180, 210]
s_cs = [15, 15, 15]

# Feed rate – 2 MFs
c_fr = [0.17, 0.25]
s_fr = [0.03, 0.03]

# Vibration – 2 MFs
c_vb = [11, 16]
s_vb = [3, 3]

# Temperature – 2 MFs
c_tp = [50, 63]
s_tp = [5, 5]


# --------------------------------------------------------
# MATRIX HELPERS (5×5 inverse)
# --------------------------------------------------------
def mat_vec(A, v):
    result = [0]*len(A)
    for i in range(len(A)):
        for j in range(len(v)):
            result[i] += A[i][j] * v[j]
    return result

def mat_mult(A, B):
    R = [[0]*len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                R[i][j] += A[i][k] * B[k][j]
    return R

# Manual Gauss-Jordan 5×5 inverse
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
# FORWARD PASS – 24 rules
# --------------------------------------------------------
def forward(x):
    cs, fr, vb, tp = x

    mu_cs = [g(cs, c_cs[i], s_cs[i]) for i in range(3)]
    mu_fr = [g(fr, c_fr[i], s_fr[i]) for i in range(2)]
    mu_vb = [g(vb, c_vb[i], s_vb[i]) for i in range(2)]
    mu_tp = [g(tp, c_tp[i], s_tp[i]) for i in range(2)]

    w = []
    for a in range(3):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    w.append(mu_cs[a] * mu_fr[b] * mu_vb[c] * mu_tp[d])

    w_sum = sum(w)
    if w_sum == 0:
        return w
    return [val / w_sum for val in w]


# --------------------------------------------------------
# TRAIN CONSEQUENTS – 24 rules, each (p1,p2,p3,p4,b)
# --------------------------------------------------------
def train_consequents(X, y):
    params = [[0]*5 for _ in range(24)]
    W_all = [forward(X[i]) for i in range(N)]

    for r in range(24):

        XtWX = [[0]*5 for _ in range(5)]
        XtWy = [0]*5

        for n in range(N):
            cs, fr, vb, tp = X[n]
            row = [cs, fr, vb, tp, 1]
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
# PREDICT
# --------------------------------------------------------
def predict(x, params):
    w = forward(x)
    f = []

    for r in range(24):
        p1, p2, p3, p4, b = params[r]
        cs, fr, vb, tp = x
        f.append(p1*cs + p2*fr + p3*vb + p4*tp + b)

    return sum(w[i] * f[i] for i in range(24))


# --------------------------------------------------------
# TRAIN + TEST
# --------------------------------------------------------
params = train_consequents(X, y)

preds = [predict(X[i], params) for i in range(N)]
rmse = math.sqrt(sum((preds[i] - y[i])**2 for i in range(N)) / N)

print("\n--- ANFIS TOOL WEAR RESULTS (PURE PYTHON) ---\n")
print("Parameters:")
for p in params:
    print(p)

print("\nPredictions:", [round(p, 3) for p in preds])
print("Actual     :", y)
print("RMSE:", rmse)
