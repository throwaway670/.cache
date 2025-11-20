import numpy as np
import random

def activation_function(x):
    return 1 if x >= 0 else 0

def train_singlelayer_perceptron(X, y, alpha=0.1):
    W = np.zeros(X.shape[1])
    b = 0

    for epoch in range(epochs):
        print("\nEpoch", epoch+1)
        for i in range(len(X)):
            net = np.dot(X[i], W) + b
            y_pred = activation_function(net)
            error = y[i] - y_pred

            # Weight Update Rule
            W = W + alpha * error * X[i]
            b = b + alpha * error
            print(f"Input:{X[i]}, Pred:{y_pred}, Error:{error}, Weights:{W}, Bias:{b}")

    return W, b

def predict_singlelayer_perceptron(X, W, b):
    for x in X:
        net = np.dot(x, W) + b
        print(x, " -> ", activation_function(net))
        
def binarystep(x, threshold=0.5):
    return 1 if x >= threshold else 0
def bipolarstep(x, threshold=0.0):
    return 1 if x >= threshold else -1
def binarysigmoid(x, _lambda=0.5):
    return 1/(1 + np.exp(-_lambda * x))
def bipolarsigmoid(x, _lambda=0.5):
    return (1 - np.exp(-_lambda * x)) / (1 + np.exp(-_lambda * x))
def ramp(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return x

def mp_neuron(weights, inputs, threshold):
    total = sum(w * x for w, x in zip(weights, inputs))
    return 1 if total >= threshold else 0

def and_mp_neuron():
    X = [[0,0],[0,1],[1,0],[1,1]]
    y = [0,0,0,1]
    print("\nTrained AND Neuron:")
    for inputs in X:
        print(f"Input: {inputs}, Output: {mp_neuron(weights=[1,1], inputs=inputs, threshold=2)}")
        
def andnot_mp_neuron():
    X = [[0,0],[0,1],[1,0],[1,1]]
    y = [0,0,1,0]
    print("\nTrained AND NOT Neuron:")
    for inputs in X:
        print(f"Input: {inputs}, Output: {mp_neuron(weights=[1,-1], inputs=inputs, threshold=1)}")
        
def or_mp_neuron():
    X = [[0,0],[0,1],[1,0],[1,1]]
    y = [0,1,1,1]
    print("\nTrained OR Neuron:")
    for inputs in X:
        print(f"Input: {inputs}, Output: {mp_neuron(weights=[1,1], inputs=inputs, threshold=1)}")
        
def xor_mp_neuron():
    X = [[0,0],[0,1],[1,0],[1,1]]
    y = [0,1,1,0]
    print("\nTrained XOR Neuron using 2 MP Neurons:")
    for inputs in X:
        a, b = inputs
        neuron1 = mp_neuron(weights=[1,-1], inputs=[a,b], threshold=1)  # AND NOT
        neuron2 = mp_neuron(weights=[-1,1], inputs=[a,b], threshold=1)  # NOT AND 
        output = mp_neuron(weights=[1,1], inputs=[neuron1, neuron2], threshold=1)  # OR
        print(f"Input: {inputs}, Output: {output}")

class bpn:
    def __init__(self,input_size,hidden_size,output_size,learningrate = 0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learningrate = learningrate

        self.W1 = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.W2 = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))

        self.b1 = np.random.uniform(-1, 1, (1, self.hidden_size)) 
        self.b2 = np.random.uniform(-1, 1, (1, self.output_size))

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def derivativesigmoid(self,x):
        activation = self.sigmoid(x)
        return activation*(1-activation)

    def training(self,X, y,epochs = 5000):
        for i in range(epochs):
            z_in = np.dot(X, self.W1) + self.b1
            z_out = self.sigmoid(z_in)

            y_in = np.dot(z_out, self.W2) + self.b2
            y_out = self.sigmoid(y_in)
            
            delta = (y - y_out) * self.derivativesigmoid(y_in)
            dW2 = np.dot(z_out.T, delta) * self.learningrate
            db2 = np.sum(delta, axis=0, keepdims=True) * self.learningrate
            
            self.W2 += dW2
            self.b2 += db2
            
            delta_hidden = np.dot(delta, self.W2.T) * self.derivativesigmoid(z_in)
            dW1 = np.dot(X.T, delta_hidden) * self.learningrate
            db1 = np.sum(delta_hidden, axis=0, keepdims=True) * self.learningrate
            
            self.W1 += dW1
            self.b1 += db1

    def predict(self, X):
        z_out = self.sigmoid(np.dot(X, self.W1) + self.b1)
        y_out = self.sigmoid(np.dot(z_out, self.W2) + self.b2)
        y_pred = np.round(y_out)
        return y_pred
    
def main_bpn():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])  # XOR output

    bpn_model = bpn(input_size=2, hidden_size=2, output_size=1, learningrate=0.5)
    bpn_model.training(X, y, epochs=10000)

    print("\nBPN Predictions for XOR:")
    predictions = bpn_model.predict(X)
    for i, x in enumerate(X):
        print(f"Input: {x}, Predicted Output: {predictions[i][0]:.4f}")
        
class ksofm:
    def __init__(self, input_size, map_size, learning_rate=0.1, sigma=None):
        self.input_size = input_size
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.sigma = sigma if sigma else max(map_size)/2
        self.weights = np.random.rand(map_size[0], map_size[1], input_size)

    def get_winner(self, x):
        distances = np.sum((self.weights - x) ** 2, axis=2)
        winner_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return winner_idx
    
    def train(self, X, epochs=1000):
        for epoch in range(epochs):
            for x in X:
                winner_idx = self.get_winner(x)
                for i in range(self.map_size[0]):
                    for j in range(self.map_size[1]):
                        dist_to_winner = np.sum((np.array([i,j]) - np.array(winner_idx)) ** 2)
                        h = np.exp(-dist_to_winner / (2 * (self.sigma ** 2)))
                        self.weights[i,j,:] += self.learning_rate * h * (x - self.weights[i,j,:])
            self.learning_rate *= 0.5 # Decay learning rate
            self.sigma *= 0.5 # Decay sigma
            if self.learning_rate < 0.001:
                break
            
        return self.weights
    
    def predict(self, x):
        winner_idx = self.get_winner(x)
        return winner_idx
    
def main_ksofm():
    X = np.array([[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]])
    ksofm_model = ksofm(input_size=2, map_size=(3,3), learning_rate=0.5)
    ksofm_model.train(X, epochs=100)

    print("\nKSOFM Weights after Training:")
    print(ksofm_model.weights)

    print("\nKSOFM Predictions:")
    for x in X:
        winner = ksofm_model.predict(x)
        print(f"Input: {x}, Winner Neuron: {winner}")
        
class lvq:
    def __init__(self, input_size, num_prototypes, learning_rate=0.1):
        self.input_size = input_size
        self.num_prototypes = num_prototypes
        self.learning_rate = learning_rate
        self.prototypes = np.random.rand(num_prototypes, input_size)
        self.prototype_labels = np.random.randint(0, 2, num_prototypes)  
    
    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            for i, x in enumerate(X):
                distances = np.sum((self.prototypes - x) ** 2, axis=1)
                winner_idx = np.argmin(distances)
                if self.prototype_labels[winner_idx] == y[i]:
                    self.prototypes[winner_idx] += self.learning_rate * (x - self.prototypes[winner_idx])
                else:
                    self.prototypes[winner_idx] -= self.learning_rate * (x - self.prototypes[winner_idx])
            self.learning_rate *= 0.95 
            if self.learning_rate < 0.001:
                break
            
    def predict(self, x):
        distances = np.sum((self.prototypes - x) ** 2, axis=1)
        winner_idx = np.argmin(distances)
        return self.prototype_labels[winner_idx]
    
def main_lvq():
    X = np.array([[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]])
    y = np.array([0, 0, 1, 1])  

    lvq_model = lvq(input_size=2, num_prototypes=4, learning_rate=0.5)
    lvq_model.train(X, y, epochs=100)

    print("\nLVQ Prototypes after Training:")
    print(lvq_model.prototypes)

    print("\nLVQ Predictions:")
    for x in X:
        label = lvq_model.predict(x)
        print(f"Input: {x}, Predicted Label: {label}")
        
class bam:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # Weights initialized to zeros
        self.weights = np.zeros((input_size, output_size))
        
    def _to_bipolar(self, v):
        # Convert binary (0, 1) to bipolar (-1, 1)
        return 2 * v - 1
        
    def _from_bipolar(self, v):
        # Convert bipolar (-1, 1) back to binary (0, 1)
        return (v + 1) // 2

    def train(self, X_binary, Y_binary):
        X = self._to_bipolar(X_binary)
        Y = self._to_bipolar(Y_binary)
        
        # Calculate the weight matrix W = sum(x_i * y_i^T)
        for x, y in zip(X, Y):
            # np.outer performs the x_i * y_i^T operation
            self.weights += np.outer(x, y)
    
    # Predict/Recall from X to Y (Forward Pass)
    def predict_forward(self, x_binary):
        x_bipolar = self._to_bipolar(x_binary)
        y_sum = np.dot(x_bipolar, self.weights)
        # Apply the bipolar sign activation function: sgn(v) = 1 if v >= 0, -1 if v < 0
        y_bipolar = np.where(y_sum >= 0, 1, -1)
        
        return self._from_bipolar(y_bipolar)
    
def main_bam():
    X_binary = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    Y_binary = np.array([[1, 0], [0, 1], [1, 1]])
    X_test = np.array([[1, 0, 0], [0, 0, 0], [1, 1, 0]])
    Y_test = np.array([[1, 0], [0, 1], [1, 1]])

    bam_model = bam(input_size=3, output_size=2)
    bam_model.train(X_binary, Y_binary)

    print("BAM Weights after Training (Bipolar):")
    print(bam_model.weights)
    print("---")

    print("BAM Predictions (Forward Pass X -> Y):")
    for x in X_binary:
        y_pred = bam_model.predict_forward(x)
        print(f"Input: {x} (Binary), Predicted Output: {y_pred} (Binary)")
    for x in X_test:
        y_pred = bam_model.predict_forward(x)
        print(f"Input: {x} (Binary), Predicted Output: {y_pred} (Binary)")
        
class hopfield:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
        
    def _to_bipolar(self, v):
        return 2 * v - 1
        
    def _from_bipolar(self, v):
        return (v + 1) // 2

    def train(self, patterns_binary):
        patterns = self._to_bipolar(patterns_binary)
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)

    def predict(self, pattern_binary, max_iterations=10):
        pattern = self._to_bipolar(pattern_binary)
        for _ in range(max_iterations):
            for i in range(self.size):
                net_input = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if net_input >= 0 else -1
        return self._from_bipolar(pattern)
    
def main_hopfield():
    patterns_binary = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    test_patterns_binary = np.array([[1, 0, 0, 0], [0, 1, 1, 1]])

    hopfield_model = hopfield(size=4)
    hopfield_model.train(patterns_binary)

    print("Hopfield Weights after Training (Bipolar):")
    print(hopfield_model.weights)
    print("---")

    print("Hopfield Predictions:")
    for p in patterns_binary:
        recalled = hopfield_model.predict(p)
        print(f"Input: {p} (Binary), Recalled Pattern: {recalled} (Binary)")
    for p in test_patterns_binary:
        recalled = hopfield_model.predict(p)
        print(f"Input: {p} (Binary), Recalled Pattern: {recalled} (Binary)")
        

class fuzzy_sets_n_relations:
    def __init__(self, universe):
        self.universe = universe
        self.sets = {} # Stores named fuzzy sets
    
    def add_set(self, name, membership_values):
        if not isinstance(membership_values, np.ndarray):
            membership_values = np.array(membership_values) 
            
        if not all(0 <= val <= 1 for val in membership_values):
            raise ValueError("Membership values must be between 0 and 1.")
        
        self.sets[name] = membership_values   
    
    def complement(self, set_name):
        mu = self.sets[set_name]
        return 1 - mu
    
    def union(self, set_nameA, set_nameB):
        muA = self.sets[set_nameA]
        muB = self.sets[set_nameB]
        return np.maximum(muA, muB)
    
    def intersection(self, set_nameA, set_nameB):
        muA = self.sets[set_nameA]
        muB = self.sets[set_nameB]
        return np.minimum(muA, muB)
    
    def alpha_cut(self, set_name, alpha):
        """Determines the alpha-cut (crisp set of elements where mu(x) >= alpha)."""
        mu = self.sets[set_name]
        return np.where(mu >= alpha, mu, 0)
    
    def verify_de_morgan(self, set_name_A, set_name_B):
        """Verifies the two De Morgan's Laws."""
        A_comp = self.complement(set_name_A)
        B_comp = self.complement(set_name_B)
        
        # Law 1: NOT(A U B) == NOT(A) n NOT(B)
        LHS1 = 1 - self.union(set_name_A, set_name_B)
        RHS1 = np.minimum(A_comp, B_comp)
        
        # Law 2: NOT(A n B) == NOT(A) U NOT(B)
        LHS2 = 1 - self.intersection(set_name_A, set_name_B)
        RHS2 = np.maximum(A_comp, B_comp)
        
        return {
            "Law 1 - Union": np.allclose(LHS1, RHS1),
            "Law 2 - Intersection": np.allclose(LHS2, RHS2),
            "LHS1": LHS1, "RHS1": RHS1,
            "LHS2": LHS2, "RHS2": RHS2
        }
        
    # --- Part 2: Membership Functions & Fuzzy Relations ---

    def triangular_mf(self, x, a, m, b):
        """
        Creates a triangular membership value for a point x with parameters (a, m, b).
        a: base start, m: peak (mu=1), b: base end
        """
        return np.maximum(0, np.minimum((x - a) / (m - a), (b - x) / (b - m)))
    
    def trapezoidal_mf(self, x, a, b, c, d):
        """
        Creates a trapezoidal membership value for a point x with parameters (a, b, c, d).
        (b, c) is the top (mu=1), a and d are the base ends (mu=0).
        """
        x = np.array(x) if isinstance(x, (list, tuple)) else x
        
        mu_rising = (x - a) / (b - a)
        mu_falling = (d - x) / (d - c)
        
        return np.maximum(0, np.minimum(np.minimum(mu_rising, 1), mu_falling))

    def relation_cartesian_product(self, set_name_A, set_name_B, t_norm='min'):
        """
        Computes the Cartesian product (fuzzy relation R = A x B) using a T-norm.
        Default is the 'min' T-norm.
        R(x, y) = T(mu_A(x), mu_B(y))
        """
        mu_A = self.sets[set_name_A].reshape(-1, 1) # Column vector
        mu_B = self.sets[set_name_B].reshape(1, -1) # Row vector

        if t_norm == 'min':
            R = np.minimum(mu_A, mu_B)
        elif t_norm == 'product':
            R = mu_A * mu_B
        else:
            raise ValueError("Unsupported T-norm. Use 'min' or 'product'.")
            
        return R

    def relation_composition(self, R_1, R_2, method='max_min'):
        """
        Computes the composite relation R_3 = R_1 o R_2.
        R_1: Matrix U x V, R_2: Matrix V x W
        Result R_3: Matrix U x W
        """
        if method == 'max_min':
            # R_3[i, k] = max_j { min(R_1[i, j], R_2[j, k]) }
            # Equivalent to matrix multiplication in the (max, min) algebra
            R_3 = np.zeros((R_1.shape[0], R_2.shape[1]))
            for i in range(R_1.shape[0]):
                for k in range(R_2.shape[1]):
                    R_3[i, k] = np.max(np.minimum(R_1[i, :], R_2[:, k]))
            return R_3
        
        elif method == 'max_product':
            # R_3[i, k] = max_j { R_1[i, j] * R_2[j, k] }
            # Equivalent to matrix multiplication in the (max, product) algebra
            R_3 = np.zeros((R_1.shape[0], R_2.shape[1]))
            for i in range(R_1.shape[0]):
                for k in range(R_2.shape[1]):
                    R_3[i, k] = np.max(R_1[i, :] * R_2[:, k])
            return R_3
        
        else:
            raise ValueError("Unsupported composition method. Use 'max_min' or 'max_product'.")

    # --- Inference and Defuzzification ---

    def inference(self, input_fuzzy_set, R_relation, composition_method='max_min'):
        """
        Performs fuzzy inference: B = A o R.
        B[j] = max_i { T(A[i], R[i, j]) }
        """
        A = input_fuzzy_set.reshape(1, -1) # Row vector
        
        if composition_method == 'max_min':
            # B[j] = max_i { min(A[i], R[i, j]) }
            B = np.max(np.minimum(A.T, R_relation), axis=0)
        elif composition_method == 'max_product':
            # B[j] = max_i { A[i] * R[i, j] }
            B = np.max(A.T * R_relation, axis=0)
        else:
            raise ValueError("Unsupported composition method. Use 'max_min' or 'max_product'.")
            
        return B

    def centroid_defuzzification(self, output_fuzzy_set, universe_output):
        """
        Computes the Centroid defuzzification (CoG).
        mu_B: membership values, U_output: universe elements
        """
        numerator = np.sum(output_fuzzy_set * universe_output)
        denominator = np.sum(output_fuzzy_set)
        
        if denominator == 0:
            return 0.0 # Handle case with empty fuzzy set
        return numerator / denominator

    def weighted_average_defuzzification(self, output_fuzzy_set, universe_output):
        """
        Computes the Weighted Average defuzzification (for symmetric MF centers).
        mu_B: membership values, U_output: universe elements (or centers)
        """
        numerator = np.sum(output_fuzzy_set * universe_output)
        denominator = np.sum(output_fuzzy_set)
        
        if denominator == 0:
            return 0.0
        return numerator / denominator
    
def main_fuzzy():
    print("## 🤖 Problem A1: Fuzzy Set Operations")
    
    # 1. Define the Universe of Discourse (U) and the Fuzzy Sets
    U = np.arange(1, 11) # {1, 2, ..., 10}
    A_mu = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2]
    B_mu = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]
    
    ft = fuzzy_sets_n_relations(universe=U)
    ft.add_set("A", A_mu)
    ft.add_set("B", B_mu)
    
    # 2. Compute A U B, A n B, and A' (A complement)
    A_union_B = ft.union("A", "B")
    A_intersection_B = ft.intersection("A", "B")
    A_complement = ft.complement("A")
    
    # 3. Verify De Morgan's Laws
    de_morgan_results = ft.verify_de_morgan("A", "B")
    
    # 4. Tabular Results
    print("\n### 📊 Tabular Results")
    results = {
        "x (U)": U,
        "mu_A(x)": ft.sets["A"],
        "mu_B(x)": ft.sets["B"],
        "A U B (max)": A_union_B,
        "A n B (min)": A_intersection_B,
        "A' (1-mu_A)": A_complement
    }
    
    table_data = list(results.values())
    headers = list(results.keys())
    
    # Printing a simple table
    print(" | ".join(f"{h:<10}" for h in headers))
    print("-" * (len(headers) * 13))
    for row in zip(*table_data):
        print(" | ".join(f"{v:<10.1f}" if isinstance(v, float) else f"{v:<10}" for v in row))
        
    print("\n### ⚖️ De Morgan's Laws Verification")
    print(f"Law 1 (NOT(A U B) == NOT(A) n NOT(B)): {de_morgan_results['Law 1 - Union']}")
    print(f"Law 2 (NOT(A n B) == NOT(A) U NOT(B)): {de_morgan_results['Law 2 - Intersection']}")
    
# main_fuzzy()

class fuzzy_logic_controller:
    """
    A flexible Fuzzy Logic Controller supporting Mamdani inference,
    Triangular/Trapezoidal MFs, and Centroid/Max-of-Min Defuzzification.
    """

    def __init__(self):
        self.inputs = {}    # Stores {'Name': {'range': (min, max), 'sets': {label: [MF_params], ...}}}
        self.output = None  # Stores {'Name': {'range': (min, max), 'sets': {label: [MF_params], ...}, 'universe': array}}
        self.rules = []     # List of rules: [(['Input1_Set', 'Input2_Set', ...], 'Output_Set'), ...]

    # --- 1. Membership Function Generation ---

    def _triangular_mf(self, x, a, m, b):
        """Standard Triangular Membership Function."""
        return np.maximum(0, np.minimum((x - a) / (m - a), (b - x) / (b - m)))

    def _trapezoidal_mf(self, x, a, b, c, d):
        """Standard Trapezoidal Membership Function."""
        mu_rising = (x - a) / (b - a)
        mu_falling = (d - x) / (d - c)
        return np.maximum(0, np.minimum(np.minimum(mu_rising, 1), mu_falling))

    def define_input(self, name, range_tuple, sets_dict):
        """Defines an input variable with its range and linguistic sets/MFs."""
        self.inputs[name] = {'range': range_tuple, 'sets': sets_dict}

    def define_output(self, name, range_tuple, num_points, sets_dict):
        """Defines the output variable with its universe for defuzzification."""
        min_val, max_val = range_tuple
        self.output = {
            'name': name,
            'range': range_tuple,
            'universe': np.linspace(min_val, max_val, num_points),
            'sets': sets_dict
        }

    # --- 2. Rule Definition ---

    def add_rule(self, antecedents, consequent):
        """
        Adds a rule. antecedents is a tuple/list of (InputName_SetLabel, ...)
        consequent is the OutputSetLabel.
        Example: (['e_NB', 'dx_dt_PB'], 'alpha_PS')
        """
        if len(antecedents) != len(self.inputs):
            raise ValueError("Rule antecedents must match the number of inputs.")
        self.rules.append((antecedents, consequent))

    # --- 3. Fuzzification ---

    def _fuzzify(self, input_values):
        """Converts crisp inputs into fuzzy membership degrees."""
        fuzzified_inputs = {}
        
        # Iterate through input variables (e.g., 'Error', 'Velocity')
        for idx, (name, input_data) in enumerate(self.inputs.items()):
            crisp_value = input_values[name]
            fuzzified_inputs[name] = {}
            
            # Iterate through sets (e.g., 'NB', 'Z', 'PB') for that variable
            for label, params in input_data['sets'].items():
                mf_type = params[0]
                mf_params = params[1:]
                
                if mf_type == 'tri':
                    mu = self._triangular_mf(crisp_value, *mf_params)
                elif mf_type == 'trap':
                    mu = self._trapezoidal_mf(crisp_value, *mf_params)
                else:
                    raise ValueError(f"Unknown MF type: {mf_type}")
                
                fuzzified_inputs[name][label] = mu
                
        return fuzzified_inputs

    # --- 4. Inference (Mamdani) ---

    def _inference(self, fuzzified_inputs, composition_method='max_min'):
        """
        Performs Mamdani inference and aggregation.
        Returns the aggregated output fuzzy set (mu_aggregated).
        """
        
        output_universe = self.output['universe']
        mu_aggregated = np.zeros_like(output_universe, dtype=float)

        for antecedents, consequent_label in self.rules:
            # 1. Rule Firing Strength (T-norm: MIN for standard Mamdani)
            # Combine all input memberships using the MIN operator (t-norm)
            firing_strengths = []
            for input_name_label in antecedents:
                input_name, set_label = input_name_label.split('_', 1)
                strength = fuzzified_inputs[input_name][set_label]
                firing_strengths.append(strength)
            
            alpha_k = np.min(firing_strengths) # Rule's firing strength

            # 2. Implication (T-norm: MIN or PRODUCT)
            # Clip (min) or Scale (product) the output MF
            
            output_set_params = self.output['sets'][consequent_label]
            mf_type = output_set_params[0]
            mf_params = output_set_params[1:]
            
            # Calculate the full MF for the consequent set
            if mf_type == 'tri':
                mu_consequent = self._triangular_mf(output_universe, *mf_params)
            elif mf_type == 'trap':
                mu_consequent = self._trapezoidal_mf(output_universe, *mf_params)
            else:
                raise ValueError("Unknown MF type in output.")
            
            # Implication (Clip/Scale)
            if composition_method == 'max_min':
                mu_implied = np.minimum(alpha_k, mu_consequent)
            elif composition_method == 'max_product':
                mu_implied = alpha_k * mu_consequent
            else:
                raise ValueError("Unsupported composition method.")

            # 3. Aggregation (T-conorm: MAX)
            mu_aggregated = np.maximum(mu_aggregated, mu_implied)
            
        return mu_aggregated

    # --- 5. Defuzzification ---

    def _defuzzify(self, mu_aggregated, method='centroid'):
        """Converts the aggregated fuzzy set into a crisp value."""
        
        U_out = self.output['universe']
        
        if method == 'centroid':
            # Centroid (Center of Gravity - CoG)
            # Applicable for A1, A2, A3, B1, B2, B3
            numerator = np.sum(mu_aggregated * U_out)
            denominator = np.sum(mu_aggregated)
            if denominator == 0:
                return 0.0
            return numerator / denominator
        
        elif method == 'max_of_min':
            # Max-of-Min (MoM) / Max Membership
            # Applicable for B4
            max_mu = np.max(mu_aggregated)
            if max_mu == 0:
                return 0.0
            
            # Find all universe values where membership is equal to max_mu
            max_indices = np.where(mu_aggregated == max_mu)[0]
            max_values = U_out[max_indices]
            
            # Return the arithmetic mean of those values
            return np.mean(max_values)
        
        else:
            raise ValueError("Unsupported defuzzification method.")

    # --- 6. Main Control Loop Execution ---

    def run_controller(self, input_values, composition_method='max_min', defuzzification_method='centroid'):
        """
        Executes the FLC for one set of crisp inputs.
        input_values is a dictionary: {'Input1_Name': value, 'Input2_Name': value, ...}
        """
        
        # 1. Fuzzification
        fuzzified_inputs = self._fuzzify(input_values)
        
        # 2. Inference & Aggregation
        mu_aggregated = self._inference(fuzzified_inputs, composition_method)
        
        # 3. Defuzzification
        crisp_output = self._defuzzify(mu_aggregated, defuzzification_method)
        
        return crisp_output, mu_aggregated
    
def configure_A1_FLC():
    flc = fuzzy_logic_controller()

    # 1. Define Input Variables (e.g., NB, NS, Z, PS, PB using Trapezoidal MFs)
    
    # Input 1: Position Error (e) - Range: -3 to +3 cm (e.g., simplified for test)
    flc.define_input(
        name='Error',
        range_tuple=(-3, 3),
        sets_dict={
            'NB': ('trap', -4, -3, -2, -1),   # Negative Big
            'NS': ('trap', -2, -1, -0.5, 0),  # Negative Small
            'Z':  ('tri', -0.5, 0, 0.5),     # Zero
            'PS': ('trap', 0, 0.5, 1, 2),    # Positive Small
            'PB': ('trap', 1, 2, 3, 4)       # Positive Big
        }
    )

    # Input 2: Velocity (dx_dt) - Range: -10 to +10 cm/s (e.g., simplified)
    # Reusing the same linguistic labels but with different parameters
    flc.define_input(
        name='Velocity',
        range_tuple=(-10, 10),
        sets_dict={
            'NB': ('trap', -12, -10, -5, -2),
            'NS': ('trap', -5, -2, -1, 0),
            'Z':  ('tri', -1, 0, 1),
            'PS': ('trap', 0, 1, 2, 5),
            'PB': ('trap', 2, 5, 10, 12)
        }
    )

    # 2. Define Output Variable (Beam Angle Alpha) - Range: -20 to +20 degrees
    # Output MFs often use the same linguistic terms as inputs in control FLCs
    flc.define_output(
        name='Alpha',
        range_tuple=(-20, 20),
        num_points=100, # Discretization for defuzzification
        sets_dict={
            'NB': ('trap', -25, -20, -15, -10),
            'NS': ('trap', -15, -10, -5, 0),
            'Z':  ('tri', -5, 0, 5),
            'PS': ('trap', 0, 5, 10, 15),
            'PB': ('trap', 10, 15, 20, 25)
        }
    )

    # 3. Define Rules (Example Rule Base: 5x5 = 25 rules)
    # The rule is IF Error is X AND Velocity is Y THEN Alpha is Z
    # Rules should be added using tuples of (Error_Set, Velocity_Set, ..., Output_Set)
    # Example Rules (a small subset for brevity):
    flc.add_rule(('Error_NB', 'Velocity_NB'), 'NB')
    flc.add_rule(('Error_NB', 'Velocity_Z'), 'NB')
    flc.add_rule(('Error_Z',  'Velocity_PB'), 'NS')
    flc.add_rule(('Error_PB', 'Velocity_Z'), 'PB')
    flc.add_rule(('Error_PS', 'Velocity_NS'), 'Z')
    flc.add_rule(('Error_Z',  'Velocity_Z'), 'Z')
    flc.add_rule(('Error_PB', 'Velocity_PB'), 'Z') # Dampening high error & high velocity

    return flc

# --- Example Execution ---

def run_A1_example():
    flc_model = configure_A1_FLC()
    
    # Test Condition: Error is 1.5 cm (Positive Small), Velocity is -3 cm/s (Negative Small)
    input_test = {'Error': 1.5, 'Velocity': -3.0}
    
    crisp_alpha, mu_agg = flc_model.run_controller(
        input_test, 
        composition_method='max_min', # Required for Mamdani
        defuzzification_method='centroid' # Required for A1
    )

    print("--- FLC Simulation (Ball-and-Beam) ---")
    print(f"Input Error: {input_test['Error']} cm")
    print(f"Input Velocity: {input_test['Velocity']} cm/s")
    print(f"Crisp Output (Beam Angle Alpha): {crisp_alpha:.4f} degrees")
    print("\nInterpretation: The beam is tilted by this angle to counteract the error.")
    
# run_A1_example()

