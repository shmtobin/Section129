import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# ------------------------------
# Helper functions
# ------------------------------
def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def softmax(z):
    # z can be 2D (for MLP) or 1D (for CNN output)
    if z.ndim == 2:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-8)
    else:
        exp_z = np.exp(z - np.max(z))
        return exp_z / (np.sum(exp_z) + 1e-8)

def cross_entropy_loss(y_pred, y_true, eps=1e-8):
    m = y_pred.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + eps)) / m
    return loss

# ------------------------------
# Data Loading and Preprocessing
# ------------------------------
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float32) / 255.0  # Normalize pixel values
y = mnist.target.astype(np.int64)
X_cnn = X.reshape(-1, 28, 28)              # For CNN: reshape to 28x28 images
X_mlp = X.copy()                          # For MLP: use flattened images
y_onehot = one_hot_encode(y, 10)

# Train-test split (80-20)
X_mlp_train, X_mlp_test, y_train, y_test, y_train_oh, y_test_oh = train_test_split(
    X_mlp, y, y_onehot, test_size=0.2, random_state=42
)
X_cnn_train, X_cnn_test, _, _, _, _ = train_test_split(
    X_cnn, y, y_onehot, test_size=0.2, random_state=42
)

print("MNIST data loaded and preprocessed.")

# ------------------------------
# Define MLPClassifier (using softmax output)
# ------------------------------
class MLPClassifier:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.weights = []
        self.biases = []
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(layer_dims) - 1):
            # He initialization for hidden layers
            self.weights.append(np.random.randn(layer_dims[i], layer_dims[i+1]) * np.sqrt(2. / layer_dims[i]))
            self.biases.append(np.zeros((1, layer_dims[i+1])))
            
    def forward(self, X):
        self.cache = {}
        A = X
        self.cache['A0'] = A
        L = len(self.weights)
        for i in range(L - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            self.cache[f'Z{i+1}'] = Z
            A = np.maximum(0, Z)  # ReLU activation
            self.cache[f'A{i+1}'] = A
        ZL = np.dot(A, self.weights[-1]) + self.biases[-1]
        self.cache[f'Z{L}'] = ZL
        A_final = softmax(ZL)
        self.cache[f'A{L}'] = A_final
        return A_final
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        L = len(self.weights)
        dZ = self.cache[f'A{L}'] - y  # Derivative for softmax cross-entropy
        for l in reversed(range(L)):
            A_prev = self.cache[f'A{l}']
            dW = np.dot(A_prev.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            # Update parameters with a lower learning rate
            self.weights[l] -= learning_rate * dW
            self.biases[l] -= learning_rate * db
            if l > 0:
                Z_prev = self.cache[f'Z{l}']
                dA_prev = np.dot(dZ, self.weights[l].T)
                dZ = dA_prev * (Z_prev > 0)  # ReLU derivative

# ------------------------------
# Define CNN components and CNNClassifier
# ------------------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)
    
    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                im_region = image[i:(i+self.filter_size), j:(j+self.filter_size)]
                yield im_region, i, j
    
    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output_h = h - self.filter_size + 1
        output_w = w - self.filter_size + 1
        output = np.zeros((output_h, output_w, self.num_filters))
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output
    
    def backward(self, d_out, learn_rate):
        d_filters = np.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_filters[f] += d_out[i, j, f] * im_region
        self.filters -= learn_rate * d_filters

class FCLayer:
    def __init__(self, input_len, output_len):
        self.weights = np.random.randn(input_len, output_len) / np.sqrt(input_len)
        self.biases = np.zeros(output_len)
    
    def forward(self, input):
        self.last_input = input
        return np.dot(input, self.weights) + self.biases
    
    def backward(self, d_out, learn_rate):
        dW = np.outer(self.last_input, d_out)
        db = d_out
        d_input = np.dot(d_out, self.weights.T)
        self.weights -= learn_rate * dW
        self.biases -= learn_rate * db
        return d_input

class CNNClassifier:
    def __init__(self, num_filters, filter_size, input_shape, num_classes, activation='relu'):
        self.conv = ConvLayer(num_filters, filter_size)
        conv_out_shape = (input_shape[0] - filter_size + 1, input_shape[1] - filter_size + 1)
        fc_input_len = conv_out_shape[0] * conv_out_shape[1] * num_filters
        self.fc = FCLayer(fc_input_len, num_classes)
        self.activation = relu if activation=='relu' else sigmoid
        self.activation_derivative = relu_derivative if activation=='relu' else sigmoid_derivative
    
    def forward(self, image):
        self.last_input = image
        self.conv_out = self.conv.forward(image)
        self.activated = self.activation(self.conv_out)
        self.fc_input = self.activated.flatten()
        self.fc_out = np.dot(self.fc_input, self.fc.weights) + self.fc.biases
        self.probs = softmax(self.fc_out)
        return self.probs
    
    def backward(self, y, learn_rate):
        d_fc = self.probs - y  # Derivative for softmax cross-entropy
        dW = np.outer(self.fc_input, d_fc)
        db = d_fc
        d_fc_input = np.dot(d_fc, self.fc.weights.T)
        self.fc.weights -= learn_rate * dW
        self.fc.biases -= learn_rate * db
        d_activated = d_fc_input.reshape(self.activated.shape)
        d_conv = d_activated * self.activation_derivative(self.conv_out)
        d_filters = np.zeros_like(self.conv.filters)
        for im_region, i, j in self.conv.iterate_regions(self.last_input):
            for f in range(self.conv.num_filters):
                d_filters[f] += d_conv[i, j, f] * im_region
        self.conv.filters -= learn_rate * d_filters

# ------------------------------
# Training parameters for MLP
# ------------------------------
mlp_epochs = 20
mlp_lr = 0.01
batch_size = 128

print("\nTraining MLP...")
mlp = MLPClassifier(input_dim=784, hidden_dims=[128, 64], output_dim=10)
mlp_train_losses = []
mlp_test_losses = []
num_train = X_mlp_train.shape[0]

for epoch in range(mlp_epochs):
    # Shuffle training data each epoch
    indices = np.arange(num_train)
    np.random.shuffle(indices)
    X_mlp_train_shuffled = X_mlp_train[indices]
    y_train_oh_shuffled = y_train_oh[indices]
    
    epoch_loss = 0.0
    num_batches = int(np.ceil(num_train / batch_size))
    for b in range(num_batches):
        start = b * batch_size
        end = start + batch_size
        X_batch = X_mlp_train_shuffled[start:end]
        y_batch = y_train_oh_shuffled[start:end]
        
        y_pred_batch = mlp.forward(X_batch)
        batch_loss = cross_entropy_loss(y_pred_batch, y_batch)
        epoch_loss += batch_loss
        mlp.backward(X_batch, y_batch, learning_rate=mlp_lr)
    epoch_loss /= num_batches
    mlp_train_losses.append(epoch_loss)
    
    y_pred_test = mlp.forward(X_mlp_test)
    test_loss = cross_entropy_loss(y_pred_test, y_test_oh)
    mlp_test_losses.append(test_loss)
    
    train_acc = accuracy_score(y_train, np.argmax(mlp.forward(X_mlp_train), axis=1))
    test_acc = accuracy_score(y_test, np.argmax(y_pred_test, axis=1))
    print(f"Epoch {epoch+1}/{mlp_epochs} - Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

mlp_preds = np.argmax(mlp.forward(X_mlp_test), axis=1)
cm_mlp = confusion_matrix(y_test, mlp_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm_mlp, annot=True, fmt="d", cmap="Blues")
plt.title("MLP Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(f'plots/cnn_confus_p3.png')
plt.show()

plt.figure()
plt.plot(range(1, mlp_epochs+1), mlp_train_losses, label="Train Loss")
plt.plot(range(1, mlp_epochs+1), mlp_test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Convergence")
plt.legend()
plt.savefig(f'plots/mlp_convergence_p3.png')
plt.show()

# ------------------------------
# Training parameters for CNN
# ------------------------------
# Use a smaller subset to speed up training (stochastic updates are slow)
subset_size = 5000
X_cnn_train_subset = X_cnn_train[:subset_size]
y_train_oh_subset = y_train_oh[:subset_size]
cnn_epochs = 5
cnn_lr = 0.001

print("\nTraining CNN (stochastic updates on a subset)...")
cnn = CNNClassifier(num_filters=8, filter_size=3, input_shape=(28, 28), num_classes=10, activation='relu')
cnn_train_losses = []
cnn_test_losses = []

for epoch in range(cnn_epochs):
    epoch_loss = 0.0
    for i in range(X_cnn_train_subset.shape[0]):
        image = X_cnn_train_subset[i]
        label = y_train_oh_subset[i]
        probs = cnn.forward(image)
        loss = -np.sum(label * np.log(probs + 1e-8))
        epoch_loss += loss
        cnn.backward(label, learn_rate=cnn_lr)
    epoch_loss /= X_cnn_train_subset.shape[0]
    cnn_train_losses.append(epoch_loss)
    
    test_loss = 0.0
    preds = []
    for i in range(X_cnn_test.shape[0]):
        image = X_cnn_test[i]
        label = y_test_oh[i]
        probs = cnn.forward(image)
        loss = -np.sum(label * np.log(probs + 1e-8))
        test_loss += loss
        preds.append(np.argmax(probs))
    test_loss /= X_cnn_test.shape[0]
    cnn_test_losses.append(test_loss)
    test_acc = accuracy_score(y_test, preds)
    print(f"Epoch {epoch+1}/{cnn_epochs} - Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

cnn_preds = []
for i in range(X_cnn_test.shape[0]):
    probs = cnn.forward(X_cnn_test[i])
    cnn_preds.append(np.argmax(probs))
cm_cnn = confusion_matrix(y_test, cnn_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Greens")
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(f'plots/cnn_confus_p3.png')
plt.show()

plt.figure()
plt.plot(range(1, cnn_epochs+1), cnn_train_losses, label="Train Loss")
plt.plot(range(1, cnn_epochs+1), cnn_test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Convergence")
plt.legend()
plt.savefig(f'plots/cnn_convergence_p3.png')
plt.show()