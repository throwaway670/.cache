# anfis_plain.py
import numpy as np

# Data: columns: Temperature, Humidity, Load(t-1), Load(t)
data = np.array([
    [30,60,220,230],
    [32,55,230,245],
    [35,50,245,260],
    [29,65,210,220],
    [28,70,205,210],
    [33,58,250,265],
    [36,48,260,275],
    [31,62,240,250],
    [27,72,200,205],
    [34,52,255,268],
], dtype=float)

# --- SELECT INPUTS ---
# For 2-input ANFIS we choose Temperature and Load(t-1)
X = data[:, [0,2]]   # shape (10,2)
y = data[:, 3]       # shape (10,)

# normalize inputs for stability
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
Xn = (X - X_mean) / X_std

n_samples, n_inputs = Xn.shape
n_mfs_per_input = 2
n_rules = n_mfs_per_input ** n_inputs

# initialize Gaussian MF parameters (centers and sigmas)
rng = np.random.RandomState(42)
centers = rng.normal(0,1,(n_inputs, n_mfs_per_input))
sigmas  = np.abs(rng.normal(1,0.5,(n_inputs, n_mfs_per_input))) + 0.5

# consequents for first-order Sugeno: y_rule = p1*x1 + p2*x2 + p0 (bias)
consequents = rng.normal(0,1,(n_rules, n_inputs+1))

def gaussian(x, c, s):
    return np.exp(-0.5 * ((x - c)/s)**2)

def forward(xn, centers, sigmas, consequents):
    # membership degrees
    mu = np.zeros((n_inputs, n_mfs_per_input))
    for i in range(n_inputs):
        for j in range(n_mfs_per_input):
            mu[i,j] = gaussian(xn[i], centers[i,j], sigmas[i,j])
    # rule firing strengths (product of chosen MFs)
    fs = np.ones(n_rules)
    rule_mf_indices = []
    for r in range(n_rules):
        idx = []
        tmp = r
        for i in range(n_inputs):
            mf_idx = tmp % n_mfs_per_input
            tmp //= n_mfs_per_input
            idx.append(mf_idx)
            fs[r] *= mu[i, mf_idx]
        rule_mf_indices.append(idx)
    # normalize weights
    if fs.sum() == 0:
        w = np.ones_like(fs) / len(fs)
    else:
        w = fs / fs.sum()
    # compute rule outputs (linear consequents)
    x_aug = np.append(xn, 1.0)
    rule_outs = consequents.dot(x_aug)
    y_hat = (w * rule_outs).sum()
    return y_hat, mu, fs, w, rule_mf_indices, rule_outs

# Training via gradient descent (simpler approach - all params by gradient)
lr_premise = 0.01
lr_conseq  = 0.1
epochs = 5000

for epoch in range(epochs):
    total_loss = 0.0
    grad_centers = np.zeros_like(centers)
    grad_sigmas = np.zeros_like(sigmas)
    grad_consequents = np.zeros_like(consequents)

    for i in range(n_samples):
        xn = Xn[i]
        yi = y[i]
        y_hat, mu, fs, w, rule_idx, rule_outs = forward(xn, centers, sigmas, consequents)
        e = y_hat - yi
        total_loss += e*e

        # gradients wrt consequents approx: dL/dp ~ 2*e * w * x_aug
        x_aug = np.append(xn, 1.0)
        for r in range(n_rules):
            grad_consequents[r] += 2*e * w[r] * x_aug

        # (chain-rule) compute gradients for premise parameters (centers, sigmas)
        fs_sum = fs.sum()
        if fs_sum == 0:
            continue
        # dw/d fs (matrix)
        dw_dfs = np.zeros((n_rules, n_rules))
        for r in range(n_rules):
            for k in range(n_rules):
                if r == k:
                    dw_dfs[r,k] = (fs_sum - fs[r]) / (fs_sum**2)
                else:
                    dw_dfs[r,k] = -fs[r] / (fs_sum**2)

        for p in range(n_inputs):
            for j in range(n_mfs_per_input):
                # dfs/dmu
                dfs_dmu = np.zeros(n_rules)
                for r in range(n_rules):
                    mf_choice = rule_idx[r][p]
                    if mu[p,j] == 0:
                        dfs_dmu[r] = 0.0
                    else:
                        if mf_choice == j:
                            dfs_dmu[r] = fs[r] / (mu[p,j] + 1e-9)
                        else:
                            dfs_dmu[r] = 0.0
                # mu derivatives wrt center/sigma
                diff = (xn[p] - centers[p,j]) / (sigmas[p,j] + 1e-9)
                mu_val = mu[p,j]
                dmu_dc = mu_val * (diff) / (sigmas[p,j] + 1e-9)
                dmu_ds = mu_val * (diff**2) / (sigmas[p,j] + 1e-9)

                # dy/dmu = sum_r ( sum_k dw_r/d fs_k * dfs_k/dmu ) * rule_out_r
                dy_dmu = 0.0
                for r in range(n_rules):
                    term = 0.0
                    for k in range(n_rules):
                        term += dw_dfs[r,k] * dfs_dmu[k]
                    dy_dmu += term * rule_outs[r]

                grad_centers[p,j] += 2*e * dy_dmu * dmu_dc
                grad_sigmas[p,j] += 2*e * dy_dmu * dmu_ds

    # average gradients and update
    grad_consequents /= n_samples
    grad_centers /= n_samples
    grad_sigmas /= n_samples

    consequents -= lr_conseq * grad_consequents
    centers -= lr_premise * grad_centers
    sigmas  -= lr_premise * grad_sigmas

    if epoch % 500 == 0 or epoch == epochs-1:
        rmse = np.sqrt(total_loss / n_samples)
        print(f"Epoch {epoch}: RMSE={rmse:.4f}")

# final predictions
preds = []
for i in range(n_samples):
    y_hat, *_ = forward(Xn[i], centers, sigmas, consequents)
    preds.append(y_hat)
preds = np.array(preds)
rmse = np.sqrt(np.mean((preds - y)**2))
print("Final RMSE:", rmse)
print("Preds:", np.round(preds,2))
print("Actual:", y)
