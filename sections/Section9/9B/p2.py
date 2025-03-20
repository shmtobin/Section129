# ---------------------------- 
# Part a
# ----------------------------

import matplotlib.pyplot as plt
import networkx as nx

def draw_cnn_graph():
    # Create a directed
    G = nx.DiGraph()

    # Add nodes representing the layers and operations
    G.add_nodes_from([
        'Input Layer', 'Conv1', 'Activation1', 'Pooling1', 
        'Conv2', 'Activation2', 'Pooling2', 'Fully Connected', 'Output Layer'
    ])
    
    # Add edges representing the flow of data through layers
    G.add_edges_from([
        ('Input Layer', 'Conv1'),
        ('Conv1', 'Activation1'),
        ('Activation1', 'Pooling1'),
        ('Pooling1', 'Conv2'),
        ('Conv2', 'Activation2'),
        ('Activation2', 'Pooling2'),
        ('Pooling2', 'Fully Connected'),
        ('Fully Connected', 'Output Layer')
    ])

    # Define positions for visualization
    pos = {
        'Input Layer': (0, 3),
        'Conv1': (1, 3),
        'Activation1': (2, 3),
        'Pooling1': (3, 3),
        'Conv2': (1, 2),
        'Activation2': (2, 2),
        'Pooling2': (3, 2),
        'Fully Connected': (2, 1),
        'Output Layer': (3, 1)
    }
    
    # Draw the graph
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
    plt.title("Forward Computational Graph for a 3-layer CNN")
    plt.savefig(f'plots/cnn_graph_p2.png', dpi=300)
    plt.show()

# Call the function to draw the graph
draw_cnn_graph()

# ---------------------------- 
# Part b - Fixed Version
# ----------------------------
import numpy as np

# Define the ReLU activation function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Define the Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Define a Convolutional Layer
class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        # Initialize filters with random values
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / 9

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                im_region = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output

    def backward(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
        self.filters -= learn_rate * d_L_d_filters
        return None  # No need to return anything for this simple example

# Define a Fully Connected Layer
class FCLayer:
    def __init__(self, input_len, output_len):
        self.weights = np.random.randn(input_len, output_len) / input_len
        self.biases = np.zeros(output_len)

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, d_L_d_out, learn_rate):
        d_L_d_input = np.dot(d_L_d_out, self.weights.T)
        d_L_d_weights = np.dot(self.last_input[:, np.newaxis], d_L_d_out[np.newaxis, :])
        d_L_d_biases = d_L_d_out

        self.weights -= learn_rate * d_L_d_weights
        self.biases -= learn_rate * d_L_d_biases
        return d_L_d_input.reshape(self.last_input_shape)

# Define the CNN class
class CNN:
    def __init__(self, num_filters, filter_size, input_shape, num_classes, activation='relu'):
        self.conv = ConvLayer(num_filters, filter_size)
        self.fc = FCLayer((input_shape[0] - filter_size + 1) * (input_shape[1] - filter_size + 1) * num_filters, num_classes)
        self.activation = relu if activation == 'relu' else sigmoid
        self.activation_derivative = relu_derivative if activation == 'relu' else sigmoid_derivative

    def forward(self, input):
        self.last_input = input
        self.conv_out = self.conv.forward(input)
        self.activated_out = self.activation(self.conv_out)
        fc_out = self.fc.forward(self.activated_out)
        return fc_out

    def backward(self, d_L_d_out, learn_rate):
        d_L_d_fc = self.fc.backward(d_L_d_out, learn_rate)
        d_L_d_activated = d_L_d_fc.reshape(self.activated_out.shape) * self.activation_derivative(self.conv_out)
        self.conv.backward(d_L_d_activated, learn_rate)

# Example usage:
cnn = CNN(num_filters=8, filter_size=3, input_shape=(28, 28), num_classes=10, activation='relu')
output = cnn.forward(np.random.randn(28, 28))
cnn.backward(np.random.randn(10), learn_rate=0.005)

def visualize_filters(conv_layer):
    filters = conv_layer.filters
    num_filters = filters.shape[0]
    fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))
    
    for i in range(num_filters):
        axes[i].imshow(filters[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i+1}')
    
    plt.suptitle('Convolutional Filters')
    plt.savefig(f'plots/conv_filters_p2.png', dpi=300)
    plt.show()

# Visualize the filters in the convolutional layer
visualize_filters(cnn.conv)

def visualize_feature_maps(feature_maps):
    num_feature_maps = feature_maps.shape[2]
    fig, axes = plt.subplots(1, num_feature_maps, figsize=(15, 5))
    
    for i in range(num_feature_maps):
        axes[i].imshow(feature_maps[:, :, i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Feature Map {i+1}')
    
    plt.suptitle('Feature Maps')
    plt.savefig(f'plots/feature_maps_p2.png', dpi=300)
    plt.show()

# Forward pass to get the feature maps
input_image = np.random.randn(28, 28)
conv_out = cnn.conv.forward(input_image)

# Visualize the feature maps
visualize_feature_maps(conv_out)

def visualize_activated_feature_maps(activated_feature_maps):
    num_feature_maps = activated_feature_maps.shape[2]
    fig, axes = plt.subplots(1, num_feature_maps, figsize=(15, 5))
    
    for i in range(num_feature_maps):
        axes[i].imshow(activated_feature_maps[:, :, i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Activated Feature Map {i+1}')
    
    plt.suptitle('Activated Feature Maps')
    plt.savefig(f'plots/activated_feature_maps_p2.png', dpi=300)
    plt.show()

# Forward pass to get the activated feature maps
activated_out = cnn.activation(conv_out)

# Visualize the activated feature maps
visualize_activated_feature_maps(activated_out)

def visualize_fc_output(fc_output):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(fc_output)), fc_output)
    plt.title('Fully Connected Layer Output')
    plt.xlabel('Class')
    plt.ylabel('Output Value')
    plt.savefig(f'plots/fc_output_p2.png', dpi=300)
    plt.show()

# Forward pass to get the fully connected layer output
fc_out = cnn.fc.forward(activated_out)

# Visualize the fully connected layer output
visualize_fc_output(fc_out)

def visualize_gradients(gradients):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(gradients)), gradients)
    plt.title('Gradients during Backpropagation')
    plt.xlabel('Parameter')
    plt.ylabel('Gradient Value')
    plt.savefig(f'plots/gradients_p2.png', dpi=300)
    plt.show()

# Backward pass to get the gradients
d_L_d_out = np.random.randn(10)
cnn.backward(d_L_d_out, learn_rate=0.005)

# Visualize the gradients (example for the fully connected layer)
visualize_gradients(cnn.fc.weights.flatten())