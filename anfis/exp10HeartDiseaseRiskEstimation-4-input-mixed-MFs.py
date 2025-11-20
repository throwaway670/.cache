# ---------------------------------------------------------
# ANFIS FOR HEART DISEASE RISK ESTIMATION — PURE PYTHON
# ---------------------------------------------------------
import math

# ---------------------------------------------------------
# DATASET  (Age, HR, LDL, BP, Risk)
# ---------------------------------------------------------
data = [
    [45, 78, 130, 120, 0.20],
    [52, 85, 150, 135, 0.35],
    [60, 90, 165, 145, 0.55],
    [48, 75, 140, 128, 0.28],
    [65, 95, 180, 155, 0.75],
    [58, 88, 155, 138, 0.50],
    [70, 100, 190, 165, 0.90],
    [55, 82, 145, 132, 0.40],
    [62, 92, 170, 150, 0.65],
    [50, 80, 135, 122, 0.25]
]

X = [row[:4] for row in data]
y = [row[4] for row in data]
N = len(X)

# ---------------------------------------------------------
# MEMBERSHIP FUNCTIONS (MIXED)
# ---------------------------------------------------------
def gauss(x, c, s):
    return math.exp(-0.5 * ((x - c) / s) ** 2)

# AGE (3 MFs)
c_age = [45, 55, 65]
s_age = [5, 5, 5]

# HEART RATE (2 MFs)
c_hr = [80, 95]
s_hr = [8, 8]

# LDL (2 MFs)
c_ldl = [140, 170]
s_ldl = [12, 12]

# BLOOD PRESSURE (2 MFs)
c_bp = [125, 150]
s_bp = [10, 10]


# ---------------------------------------------------------
# FORWARD PASS — 24 RULES
# ---------------------------------------------------------
def forward(x):
    age, hr, ldl, bp = x

    mu_age = [gauss(age, c_age[i], s_age[i]) for i in range(3)]
    mu_hr  = [gauss(hr,  c_hr[i],  s_hr[i])  for i in range(2)]
    mu_ldl = [gauss(ldl, c_ldl[i], s_ldl[i]) for i in range(2)]
    mu_bp  = [gauss(bp,  c_bp[i],  s_bp[i])  for i in range(2)]

    w = []
    for i in range(3):      # Age
        for j in range(2):  # HR
            for k in range(2):  # LDL
                for m in range(2):  # BP
                    w.append(mu_age[i] * mu_hr[j] * mu_ldl[k] * mu_bp[m])

    w_sum = sum(w)
    w_norm = [wi / w_sum for wi in w] if w_sum != 0 else w
    return w_norm   # length 24


# ---------------------------------------------------------
# MATRIX INVERSE (5x5) — PURE PYTHON
# ---------------------------------------------------------
def invert_matrix(A):
    n = len(A)
    aug = [A[i] + [1 if i == j else 0 for j in range(n)] for i in range(n)]

    for col in range(n):
        pivot = aug[col][col]
        if pivot == 0:
            raise ValueError("Matrix is singular")

        for j in range(col, 2*n):
            aug[col][j] /= pivot

        for i in range(n):
            if i != col:
                factor = aug[i][col]
                for j in range(col, 2*n):
                    aug[i][j] -= factor * aug[col][j]

    return [row[n:] for row in aug]


# ---------------------------------------------------------
# TRAIN CONSEQUENTS USING WEIGHTED LEAST SQUARES
# ---------------------------------------------------------
def train():
    X_aug = [[x[0], x[1], x[2], x[3], 1.0] for x in X]  # 5 inputs
    params = [[0]*5 for _ in range(24)]
    W_all = [forward(X[i]) for i in range(N)]

    for r in range(24):
        XtWX = [[0]*5 for _ in range(5)]
        XtWy = [0]*5

        for i in range(N):
            w = W_all[i][r]
            Xi = X_aug[i]

            for a in range(5):
                for b in range(5):
                    XtWX[a][b] += w * Xi[a] * Xi[b]

            for a in range(5):
                XtWy[a] += w * Xi[a] * y[i]

        inv = invert_matrix(XtWX)
        sol = [sum(inv[row][j] * XtWy[j] for j in range(5)) for row in range(5)]
        params[r] = sol

    return params


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
def predict(x, params):
    w_norm = forward(x)
    f = []

    for r in range(24):
        p1, p2, p3, p4, b = params[r]
        out = p1*x[0] + p2*x[1] + p3*x[2] + p4*x[3] + b
        f.append(out)

    return sum(f[i] * w_norm[i] for i in range(24))


# ---------------------------------------------------------
# TRAIN + TEST
# ---------------------------------------------------------
params = train()
preds = [predict(X[i], params) for i in range(N)]
rmse = math.sqrt(sum((preds[i] - y[i])**2 for i in range(N)) / N)

# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------
print("\n--- ANFIS HEART RISK (PURE PYTHON) ---\n")
print("24 Rule Parameters (p1, p2, p3, p4, b):\n")
for rule in params:
    print(rule)

print("\nPredictions:", [round(p, 3) for p in preds])
print("Actual     :", y)
print("\nRMSE =", rmse)
