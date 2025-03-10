# ---------------------------- 
# Part a
# ----------------------------

import matplotlib.pyplot as plt
import networkx as nx

def draw_cnn_graph():
    # Create a directed graph
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
    plt.show()

# Call the function to draw the graph
draw_cnn_graph()

# ---------------------------- 
# Part b - Fixed Version
# ----------------------------
# NEED TO FIX SO IT WILL RUN AAAAAAAAAAAAAAAAAAAAAAAa
import numpy as np

class CNN:
    def __init__(self, input_size, conv1_filters, conv2_filters, fc_units, output_size):
        # Initialize the CNN layers
        self.input_size = input_size
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.fc_units = fc_units
        self.output_size = output_size
        
        # Initialize weights for convolutional layers, fully connected layers, etc.
        # Conv1 (simple filter example)
        self.conv1_weights = np.random.randn(self.conv1_filters, input_size[0], 3, 3)  # [filters, height, width]
        self.conv1_bias = np.random.randn(self.conv1_filters)
        
        # Conv2 (simple filter example)
        self.conv2_weights = np.random.randn(self.conv2_filters, self.conv1_filters, 3, 3)
        self.conv2_bias = np.random.randn(self.conv2_filters)
        
        # Calculate the output size after convolutions and pooling
        # Assuming a 2x2 pooling layer after each convolution
        # Calculate the output size after convolutions and pooling
        conv1_output_size = (input_size[1] - 3 + 1) // 2  # After Conv1 + Pooling
        conv2_output_size = (conv1_output_size - 3 + 1) // 2  # After Conv2 + Pooling
        flattened_size = self.conv2_filters * conv2_output_size * conv2_output_size  # Flattened size for FC layer

        
        # Fully connected layer
        self.fc_weights = np.random.randn(self.fc_units, flattened_size)  # Adjusted for the correct flattened size
        self.fc_bias = np.random.randn(self.fc_units)
        
        # Output layer
        self.output_weights = np.random.randn(self.output_size, self.fc_units)
        self.output_bias = np.random.randn(self.output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def convolution2d(self, input_data, weights, bias):
        # Simple convolution operation (no padding, stride = 1)
        filter_height, filter_width = weights.shape[2], weights.shape[3]
        input_height, input_width = input_data.shape[1], input_data.shape[2]
        
        # Calculate output dimensions
        output_height = input_height - filter_height + 1
        output_width = input_width - filter_width + 1
        
        output = np.zeros((weights.shape[0], output_height, output_width))
        
        # Perform convolution
        for f in range(weights.shape[0]):  # Iterate over filters
            for i in range(output_height):
                for j in range(output_width):
                    output[f, i, j] = np.sum(input_data[:, i:i+filter_height, j:j+filter_width] * weights[f]) + bias[f]
                    
        return output

    def forward(self, x):
        # Forward pass through the network
        
        # Step 1: Apply first convolution
        conv1_out = self.convolution2d(x, self.conv1_weights, self.conv1_bias)
        activation1_out = self.relu(conv1_out)  # Activation function
        
        # Step 2: Apply second convolution
        conv2_out = self.convolution2d(activation1_out, self.conv2_weights, self.conv2_bias)
        activation2_out = self.relu(conv2_out)  # Activation function
        
        # Step 3: Flatten the output of Conv2 and Pooling layer
        flattened_out = activation2_out.flatten()

        # Step 4: Fully Connected Layer (FC)
        fc_out = np.dot(self.fc_weights, flattened_out) + self.fc_bias
        fc_out = self.relu(fc_out)  # Activation function for fully connected layer
        
        # Step 5: Output Layer
        output = np.dot(self.output_weights, fc_out) + self.output_bias
        output = self.sigmoid(output)  # Sigmoid activation for final output
        
        return output

# Example of how to create and use the CNN class
if __name__ == "__main__":
    # Example input size (1 sample, 28x28 image with 1 channel)
    input_data = np.random.randn(1, 28, 28)  # 28x28 image
    
    # Create a CNN instance
    cnn = CNN(input_size=(1, 28, 28), conv1_filters=8, conv2_filters=16, fc_units=64, output_size=10)
    
    # Forward pass through the CNN
    output = cnn.forward(input_data)
    print("CNN output:", output)

# ----------------------------
# End of Part b - Fixed Version
# ----------------------------
