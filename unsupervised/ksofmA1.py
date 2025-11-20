import numpy as np

def normalize(data):
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

class KSOFM:
    def __init__(self, x, y, dim, lr=0.5, sigma=1.0, epochs=100):
        self.x = x
        self.y = y
        self.dim = dim
        self.lr = lr
        self.sigma = sigma
        self.epochs = epochs
        self.weights = np.random.rand(x, y, dim)
        self.location = np.array([[i, j] for i in range(x) for j in range(y)]).reshape(x, y, 2)

    def neighborhood(self, winner, current, sigma):
        dist = np.sum((winner - current) ** 2)
        return np.exp(-dist / (2 * (sigma ** 2)))

    def train(self, data):
        data = normalize(data)
        for epoch in range(self.epochs):
            for sample in data:
                dists = np.linalg.norm(self.weights - sample, axis=2)
                winner_idx = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
                for i in range(self.x):
                    for j in range(self.y):
                        neigh_strength = self.neighborhood(np.array(winner_idx), np.array([i, j]), self.sigma)
                        lr_effective = self.lr * neigh_strength
                        self.weights[i, j] += lr_effective * (sample - self.weights[i, j])
            self.lr *= 0.9
            self.sigma *= 0.9

    def predict(self, data):
        data = normalize(data)
        preds = []
        for sample in data:
            dists = np.linalg.norm(self.weights - sample, axis=2)
            winner_idx = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
            preds.append((int(winner_idx[0]), int(winner_idx[1])))
        return preds

# Sample Data (Countries)
data = np.array([
  [55, 3.2, 45000, 85, 10],
  [20, 9.0, 30000, 70, 50],
  [10, 12.5, 15000, 40, 80],
  [75, 1.5, 50000, 90, 5],
  [25, 8.0, 20000, 60, 40],
  [65, 2.0, 48000, 88, 8]
])

model = KSOFM(2, 2, 5, lr=0.5, sigma=1.0, epochs=100)
model.train(data)
clusters = model.predict(data)
print("Country clusters:", clusters)
