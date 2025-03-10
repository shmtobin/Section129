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
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int64)
X_cnn = X.reshape(-1, 28, 28)    # For CNN: shape (num_examples, 28, 28)
X_mlp = X.copy()                # For MLP: flattened 784-d vectors
y_onehot = one_hot_encode(y, 10)

# 80-20 train-test split
X_mlp_train, X_mlp_test, y_train, y_test, y_train_oh, y_test_oh = train_test_split(
    X_mlp, y, y_onehot, test_size=0.2, random_state=42
)
X_cnn_train, X_cnn_test, _, _, _, _ = train_test_split(
    X_cnn, y, y_onehot, test_size=0.2, random_state=42
)
print("MNIST data loaded and preprocessed.")

# ------------------------------
# Define MLPClassifier (unchanged)
# ------------------------------
class MLPClassifier:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.weights = []
        self.biases = []
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(layer_dims) - 1):
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
            A = np.maximum(0, Z)
            self.cache[f'A{i+1}'] = A
        ZL = np.dot(A, self.weights[-1]) + self.biases[-1]
        self.cache[f'Z{L}'] = ZL
        A_final = softmax(ZL)
        self.cache[f'A{L}'] = A_final
        return A_final
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        L = len(self.weights)
        dZ = self.cache[f'A{L}'] - y
        for l in reversed(range(L)):
            A_prev = self.cache[f'A{l}']
            dW = np.dot(A_prev.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            self.weights[l] -= learning_rate * dW
            self.biases[l] -= learning_rate * db
            if l > 0:
                Z_prev = self.cache[f'Z{l}']
                dA_prev = np.dot(dZ, self.weights[l].T)
                dZ = dA_prev * (Z_prev > 0)

# ------------------------------
# CNN Components (Improved Version)
# ------------------------------

# Modified ConvLayer that returns d_input during backpropagation
class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)
    
    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                im_region = image[i:i+self.filter_size, j:j+self.filter_size]
                yield im_region, i, j
    
    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        out_h = h - self.filter_size + 1
        out_w = w - self.filter_size + 1
        output = np.zeros((out_h, out_w, self.num_filters))
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output
    
    def backward(self, d_out, learn_rate):
        d_filters = np.zeros(self.filters.shape)
        d_input = np.zeros_like(self.last_input)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_filters[f] += d_out[i, j, f] * im_region
                d_input[i:i+self.filter_size, j:j+self.filter_size] += self.filters[f] * d_out[i, j, f]
        self.filters -= learn_rate * d_filters
        return d_input

# Fully connected layer (unchanged)
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

# Max pooling layer (2x2)
class MaxPoolLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
    def forward(self, input):
        self.input = input
        h, w, f = input.shape
        out_h = h // self.pool_size
        out_w = w // self.pool_size
        output = np.zeros((out_h, out_w, f))
        self.mask = {}
        for i in range(out_h):
            for j in range(out_w):
                region = input[i*self.stride:i*self.stride+self.pool_size,
                               j*self.stride:j*self.stride+self.pool_size, :]
                output[i, j, :] = np.max(region, axis=(0, 1))
                mask = (region == np.max(region, axis=(0, 1), keepdims=True))
                self.mask[(i, j)] = mask
        self.out_shape = output.shape
        return output
    def backward(self, d_out):
        d_input = np.zeros_like(self.input)
        out_h, out_w, f = d_out.shape
        for i in range(out_h):
            for j in range(out_w):
                mask = self.mask[(i, j)]
                d_input[i*self.stride:i*self.stride+self.pool_size,
                        j*self.stride:j*self.stride+self.pool_size, :] += mask * d_out[i, j, :][None, None, :]
        return d_input

# Improved CNNClassifier with two conv layers and a max pooling layer
class CNNClassifierImproved:
    def __init__(self, input_shape, num_classes):
        # First conv layer: input (28,28), filter size=3, 8 filters → output: (26,26,8)
        self.conv1 = ConvLayer(num_filters=8, filter_size=3)
        self.pool1 = MaxPoolLayer(pool_size=2, stride=2)  # → output: (13,13,8)
        # Second conv layer: filter size=3, 16 filters → output: (13-3+1, 13-3+1, 16) = (11,11,16)
        self.conv2 = ConvLayer(num_filters=16, filter_size=3)
        # Fully connected layer: 11*11*16 → num_classes
        self.fc = FCLayer(11 * 11 * 16, num_classes)
    
    def forward(self, image):
        # image shape: (28,28)
        self.input = image
        self.conv1_out = self.conv1.forward(image)           # (26,26,8)
        self.act1 = np.maximum(0, self.conv1_out)              # ReLU
        self.pool1_out = self.pool1.forward(self.act1)         # (13,13,8)
        self.conv2_out = self.conv2.forward(self.pool1_out)    # (11,11,16)
        self.act2 = np.maximum(0, self.conv2_out)              # ReLU
        self.flat = self.act2.flatten()
        self.fc_out = np.dot(self.flat, self.fc.weights) + self.fc.biases
        self.probs = softmax(self.fc_out)
        return self.probs
    
    def backward(self, y, learn_rate):
        # Backprop through FC layer
        d_fc = self.probs - y  # (num_classes,)
        dW_fc = np.outer(self.flat, d_fc)
        db_fc = d_fc
        d_flat = np.dot(d_fc, self.fc.weights.T)
        self.fc.weights -= learn_rate * dW_fc
        self.fc.biases -= learn_rate * db_fc
        d_act2 = d_flat.reshape(self.act2.shape)
        # Backprop through ReLU (conv2)
        d_conv2 = d_act2 * (self.conv2_out > 0)
        d_pool1 = self.conv2.backward(d_conv2, learn_rate)
        # Backprop through max pooling
        d_act1 = self.pool1.backward(d_pool1)
        # Backprop through ReLU (conv1)
        d_conv1 = d_act1 * (self.conv1_out > 0)
        _ = self.conv1.backward(d_conv1, learn_rate)

# ------------------------------
# Training Loop for MLP (unchanged)
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
    print(f"MLP Epoch {epoch+1}/{mlp_epochs} - Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

mlp_preds = np.argmax(mlp.forward(X_mlp_test), axis=1)
cm_mlp = confusion_matrix(y_test, mlp_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm_mlp, annot=True, fmt="d", cmap="Blues")
plt.title("MLP Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(f'plots/mlp_confus.png')
plt.show()

plt.figure()
plt.plot(range(1, mlp_epochs+1), mlp_train_losses, label="Train Loss")
plt.plot(range(1, mlp_epochs+1), mlp_test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Convergence")
plt.legend()
plt.savefig(f'plots/mlp_converg.png')
plt.show()

# ------------------------------
# Training Loop for Improved CNN
# ------------------------------
cnn_epochs = 10
cnn_lr = 0.001

print("\nTraining Improved CNN...")
cnn = CNNClassifierImproved(input_shape=(28,28), num_classes=10)
cnn_train_losses = []
cnn_test_losses = []

for epoch in range(cnn_epochs):
    epoch_loss = 0.0
    for i in range(X_cnn_train.shape[0]):
        image = X_cnn_train[i]
        label = y_train_oh[i]
        probs = cnn.forward(image)
        loss = -np.sum(label * np.log(probs + 1e-8))
        epoch_loss += loss
        cnn.backward(label, learn_rate=cnn_lr)
    epoch_loss /= X_cnn_train.shape[0]
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
    print(f"Improved CNN Epoch {epoch+1}/{cnn_epochs} - Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

cnn_preds = []
for i in range(X_cnn_test.shape[0]):
    probs = cnn.forward(X_cnn_test[i])
    cnn_preds.append(np.argmax(probs))
cm_cnn = confusion_matrix(y_test, cnn_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Greens")
plt.title("Improved CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(f'plots/cnn_confus.png')
plt.show()

plt.figure()
plt.plot(range(1, cnn_epochs+1), cnn_train_losses, label="Train Loss")
plt.plot(range(1, cnn_epochs+1), cnn_test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Convergence")
plt.legend()
plt.savefig(f'plots/cnn_converg.png')
plt.show()