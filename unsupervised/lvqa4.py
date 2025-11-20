import numpy as np

class LVQ:
    def __init__(self, n_prototypes, input_dim, learning_rate=0.3, epochs=100):
        self.n_prototypes = n_prototypes
        self.input_dim = input_dim
        self.lr = learning_rate
        self.epochs = epochs
        self.prototypes = np.random.rand(n_prototypes, input_dim)
        self.prototype_labels = np.zeros(n_prototypes)

    def fit(self, X, y):
        classes = np.unique(y)
        prototypes = []
        labels = []
        for c in classes:
            candidates = X[y==c]
            prototypes.append(candidates[0])
            labels.append(c)
        self.prototypes = np.array(prototypes)
        self.prototype_labels = np.array(labels)

        for epoch in range(self.epochs):
            for i, x in enumerate(X):
                distances = np.linalg.norm(self.prototypes - x, axis=1)
                winner_idx = np.argmin(distances)
                winner_label = self.prototype_labels[winner_idx]
                if winner_label == y[i]:
                    self.prototypes[winner_idx] += self.lr * (x - self.prototypes[winner_idx])
                else:
                    self.prototypes[winner_idx] -= self.lr * (x - self.prototypes[winner_idx])

    def predict(self, X):
        preds = []
        for x in X:
            distances = np.linalg.norm(self.prototypes - x, axis=1)
            winner_idx = np.argmin(distances)
            preds.append(int(self.prototype_labels[winner_idx]))
        return preds

# Sample Data (Students)
X = np.array([
    [9.0, 95, 8, 6, 7],
    [8.0, 85, 4, 7, 9],
    [7.0, 70, 5, 5, 6],
    [6.5, 60, 3, 8, 8],
    [9.2, 90, 9, 4, 5],
    [8.5, 88, 5, 6, 7]
])
y = np.array([1, 3, 4, 3, 1, 2])  # Class labels

model = LVQ(n_prototypes=4, input_dim=5, learning_rate=0.3, epochs=100)
model.fit(X, y)
predictions = model.predict(X)
print("Student classifications:", predictions)
