# 9B P1

# ---------------------------- 
# Part a
# ----------------------------

# Computational Graph Description
# 
# The neural network consists of 4 layers with the following:
# 
# Layer 1: Input dimension = 6, Number of Neurons = 4
# Layer 2: Input dimension = 4, Number of Neurons = 3
# Layer 3: Input dimension = 3, Number of Neurons = 2
# Layer 4: Input dimension = 2, Number of Neurons = 1
#
# Forward Pass Computational Graph (Example)
# 
# X = Input
# A = Activation
# Z = Linear Transformation (weights * inputs + bias)
#
# 1) Forward Pass:
# X -> Z1 (Layer 1) -> A1 (ReLU or Sigmoid) -> Z2 (Layer 2) -> A2 (ReLU or Sigmoid) 
# -> Z3 (Layer 3) -> A3 (ReLU or Sigmoid) -> Z4 (Layer 4) -> Output
#
# 2) Backward Pass (Gradient Flow):
# Output -> Backpropagate Gradients through Z4, A3, Z3, A2, Z2, A1, Z1
# (Through Derivatives of the Activation Functions)

# Below is the Python class for the forward and backward computations.

import numpy as np

class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        for i in range(len(layer_dims) - 1):
            # Weights of the form (input_dim, output_dim)
            self.weights.append(np.random.randn(layer_dims[i], layer_dims[i+1]))
            # Biases of the form (output_dim,)
            self.biases.append(np.zeros((layer_dims[i+1],)))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        self.cache = {}  # Cache for backpropagation
        A = X
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.relu(Z)  # Using ReLU activation
            self.cache[f'Z{i+1}'] = Z
            self.cache[f'A{i+1}'] = A
        # Final output layer
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self.sigmoid(Z)  # Sigmoid activation
        self.cache[f'Z{len(self.weights)}'] = Z
        self.cache[f'A{len(self.weights)}'] = A
        return A

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]  # Number of examples
        gradients = {}
        
        # Output layer
        A_final = self.cache[f'A{len(self.weights)}']
        dA = A_final - y
        dZ = dA * A_final * (1 - A_final)  # Derivative of sigmoid
        gradients[f'dW{len(self.weights)}'] = np.dot(self.cache[f'A{len(self.weights)-1}'].T, dZ) / m
        gradients[f'db{len(self.weights)}'] = np.sum(dZ, axis=0, keepdims=True) / m

        # Backpropagate through hidden layers
        for i in range(len(self.weights)-2, -1, -1):
            dA = np.dot(dZ, self.weights[i+1].T)
            dZ = dA * (self.cache[f'A{i+1}'] > 0)  # Derivative of ReLU
            gradients[f'dW{i+1}'] = np.dot(self.cache[f'A{i}'].T, dZ) / m
            gradients[f'db{i+1}'] = np.sum(dZ, axis=0, keepdims=True) / m

        # Update parameters (Weights and Biases)
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[f'dW{i+1}']
            self.biases[i] -= learning_rate * gradients[f'db{i+1}']
        
        return gradients

# Example of how to initialize and use the MLP class:
input_dim = 6
hidden_dims = [4, 3, 2]
output_dim = 1

# Create the MLP model
mlp_model = MLP(input_dim, hidden_dims, output_dim)

# Example forward pass
X_example = np.random.randn(5, input_dim)  # 5 examples with 6 features each
y_example = np.random.randn(5, output_dim)  # 5 examples of output

# Forward pass
predictions = mlp_model.forward(X_example)

# Backward pass
gradients = mlp_model.backward(X_example, y_example)

print("Predictions:", predictions)
print("Gradients:", gradients)

# ---------------------------- 
# Part b
# ----------------------------

import numpy as np

class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        for i in range(len(layer_dims) - 1):
            # Weights of the form (input_dim, output_dim)
            self.weights.append(np.random.randn(layer_dims[i], layer_dims[i+1]))
            # Biases of the form (output_dim,)
            self.biases.append(np.zeros((layer_dims[i+1],)))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        self.cache = {}  # Cache for backpropagation
        A = X
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.relu(Z)  # Using ReLU activation
            self.cache[f'Z{i+1}'] = Z
            self.cache[f'A{i+1}'] = A
        # Final output layer
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self.sigmoid(Z)  # Sigmoid activation
        self.cache[f'Z{len(self.weights)}'] = Z
        self.cache[f'A{len(self.weights)}'] = A
        return A

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]  # Number of examples
        gradients = {}
        
        # Output layer (Sigmoid activation)
        A_final = self.cache[f'A{len(self.weights)}']
        dA = A_final - y
        dZ = dA * A_final * (1 - A_final)  # Derivative of sigmoid
        gradients[f'dW{len(self.weights)}'] = np.dot(self.cache[f'A{len(self.weights)-1}'].T, dZ) / m
        gradients[f'db{len(self.weights)}'] = np.sum(dZ, axis=0, keepdims=True) / m

        # Backpropagate through hidden layers
        for i in range(len(self.weights)-2, -1, -1):
            dA = np.dot(dZ, self.weights[i+1].T)
            dZ = dA * (self.cache[f'A{i+1}'] > 0)  # Derivative of ReLU
            gradients[f'dW{i+1}'] = np.dot(self.cache[f'A{i}'].T, dZ) / m
            gradients[f'db{i+1}'] = np.sum(dZ, axis=0, keepdims=True) / m

        # Update parameters (Weights and Biases)
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[f'dW{i+1}']
            self.biases[i] -= learning_rate * gradients[f'db{i+1}']
        
        return gradients

# Example of how to initialize and use the MLP class:
input_dim = 6
hidden_dims = [4, 3, 2]
output_dim = 1

# Create the MLP model
mlp_model = MLP(input_dim, hidden_dims, output_dim)

# Example forward pass
X_example = np.random.randn(5, input_dim)  # 5 examples with 6 features each
y_example = np.random.randn(5, output_dim)  # 5 examples of output

# Forward pass
predictions = mlp_model.forward(X_example)

# Backward pass
gradients = mlp_model.backward(X_example, y_example)

print("Predictions:", predictions)
print("Gradients:", gradients)