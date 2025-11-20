# Give me numerical on backpropogation where a dataset is given For eg col1 col2 target 1 40 0 10 70 1 5 10 1 0 5 0 Model: 2->3->1 Use bipolar sigmoidal function

import numpy as np

def bipolar_sig(x):
    return 2.0 / (1.0 + np.exp(-x)) - 1.0

def bipolar_sig_der_from_out(fx):
    return 0.5 * (1 - fx**2)

# Data (raw)
data = np.array([[1,40,0],[10,70,1],[5,10,1],[0,5,0]], dtype=float)
X = np.column_stack((data[:,0]/10.0, data[:,1]/70.0))
y = np.where(data[:,2]==0, -1.0, 1.0).reshape(-1,1)

# Initialize (same values I used)
W_ih = np.array([[ 0.2, -0.1],[ 0.4,  0.1],[-0.3,  0.2]])
b_h  = np.array([0.1, -0.2, 0.05])
W_ho = np.array([[0.3],[-0.2],[0.1]])
b_o  = np.array([0.05])
lr = 0.5

def forward(x, W_ih, b_h, W_ho, b_o):
    net_h = (W_ih @ x).flatten() + b_h
    out_h = bipolar_sig(net_h)
    net_o = (W_ho.T @ out_h.reshape(3,1)).flatten() + b_o
    out_o = bipolar_sig(net_o)[0]
    return net_h, out_h, net_o[0], out_o

# Train a few epochs
for epoch in range(20):
    mse = 0.0
    for i in range(X.shape[0]):
        x = X[i].reshape(2,1)
        t = y[i,0]
        net_h, out_h, net_o, out_o = forward(x, W_ih, b_h, W_ho, b_o)
        err = t - out_o
        mse += err**2
        d_out = err * bipolar_sig_der_from_out(out_o)
        delta_h = bipolar_sig_der_from_out(out_h) * (W_ho.flatten() * d_out)
        grad_W_ho = (out_h.reshape(3,1) * d_out).reshape(3,1)
        grad_b_o = d_out
        grad_W_ih = delta_h.reshape(3,1) @ x.reshape(1,2)
        grad_b_h = delta_h
        W_ho += lr * grad_W_ho
        b_o  += lr * grad_b_o
        W_ih += lr * grad_W_ih
        b_h  += lr * grad_b_h
    mse /= X.shape[0]
    if epoch % 5 == 0:
        print("Epoch", epoch, "MSE=", mse)

# Final predictions
for i in range(X.shape[0]):
    x = X[i].reshape(2,1)
    _, _, _, out_o = forward(x, W_ih, b_h, W_ho, b_o)
    pred_class = 1 if out_o > 0 else 0
    print("Input:", data[i,:2], "Pred(out)=", round(out_o,4), "Class=", pred_class, "Target=", int(data[i,2]))
