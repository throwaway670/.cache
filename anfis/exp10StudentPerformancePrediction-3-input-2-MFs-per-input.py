import math

# --------------------------------------------
# DATASET (pure python lists)
# --------------------------------------------
data = [
    [4, 60, 55, 48],
    [8, 80, 70, 72],
    [6, 75, 65, 68],
    [10, 85, 80, 82],
    [3, 50, 45, 40],
    [12, 90, 85, 88],
    [5, 65, 60, 55],
    [9, 82, 78, 76],
    [2, 45, 40, 35],
    [11, 88, 83, 85],
    [7, 78, 72, 70]
]

X = [row[:3] for row in data]
y = [row[3] for row in data]
N = len(X)

# --------------------------------------------
# GAUSSIAN MFS
# --------------------------------------------
c_H = [4, 10]
s_H = [2.0, 2.0]

c_A = [55, 85]
s_A = [10, 10]

c_S = [50, 80]
s_S = [10, 10]

def gauss(x, c, s):
    return math.exp(-0.5 * ((x - c) / s) ** 2)

# --------------------------------------------
# MATRIX HELPERS (4x4 only)
# --------------------------------------------
def mat_mult(A, B):
    rows = len(A)
    cols = len(B[0])
    common = len(B)
    R = [[0]*cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            for k in range(common):
                R[i][j] += A[i][k] * B[k][j]
    return R

def mat_vec(A, v):
    rows = len(A)
    cols = len(A[0])
    R = [0]*rows
    for i in range(rows):
        for j in range(cols):
            R[i] += A[i][j] * v[j]
    return R

# manual 4x4 inverse (Gauss–Jordan)
def invert_4x4(M):
    n = 4
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

# --------------------------------------------
# FORWARD PASS
# --------------------------------------------
def forward(x):
    H, A, S = x

    mu_H = [gauss(H, c_H[0], s_H[0]), gauss(H, c_H[1], s_H[1])]
    mu_A = [gauss(A, c_A[0], s_A[0]), gauss(A, c_A[1], s_A[1])]
    mu_S = [gauss(S, c_S[0], s_S[0]), gauss(S, c_S[1], s_S[1])]

    w = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                w.append(mu_H[i] * mu_A[j] * mu_S[k])

    w_sum = sum(w)
    if w_sum == 0:
        return w
    return [val / w_sum for val in w]

# --------------------------------------------
# TRAIN CONSEQUENTS (8 rules, each 4 params)
# --------------------------------------------
def train_consequents(X, y):
    params = [[0]*4 for _ in range(8)]
    W_all = [forward(X[i]) for i in range(N)]

    for r in range(8):
        # XtWX = 4x4, XtWy = 4x1
        XtWX = [[0]*4 for _ in range(4)]
        XtWy = [0]*4

        for n in range(N):
            H, A, S = X[n]
            row = [H, A, S, 1]
            w = W_all[n][r]

            # XtWX += w * row^T * row
            for i in range(4):
                for j in range(4):
                    XtWX[i][j] += w * row[i] * row[j]

            # XtWy += w * row * y
            for i in range(4):
                XtWy[i] += w * row[i] * y[n]

        inv = invert_4x4(XtWX)
        params[r] = mat_vec(inv, XtWy)

    return params

# --------------------------------------------
# PREDICT
# --------------------------------------------
def predict(x, params):
    w_norm = forward(x)
    f = []

    for r in range(8):
        p1, p2, p3, b = params[r]
        H, A, S = x
        f.append(p1*H + p2*A + p3*S + b)

    return sum(w_norm[i] * f[i] for i in range(8))

# --------------------------------------------
# TRAIN + EVALUATE
# --------------------------------------------
params = train_consequents(X, y)

preds = [predict(X[i], params) for i in range(N)]

rmse = math.sqrt(sum((preds[i] - y[i])**2 for i in range(N)) / N)

print("Trained ANFIS Parameters:")
for p in params:
    print(p)

print("\nPredictions:", [round(p, 2) for p in preds])
print("Actual     :", y)
print("RMSE:", rmse)
