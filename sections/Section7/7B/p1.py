# 7B P1

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Task 1: Sobol sequence
# ----------------------------

def sobol_2d(n):
    """
    Generate a 2D Sobol sequence using the given primitive polynomial:
    p2(x) = x^3 + x^2 + x + 1
    """
    # Parameters from the problem
    m = [1, 3, 5]  # m1, m2, m3
    v = [1/2, 3/4, 5/8]  # v2,1, v2,2, v2,3
    
    # Initialize arrays
    sobol_seq = np.zeros((n, 2))
    direction_numbers = np.zeros((len(m), n), dtype=float)
    
    # Compute direction numbers
    for i in range(len(m)):
        for j in range(n):
            direction_numbers[i, j] = v[i] / (2 ** (j + 1))
    
    # Generate Sobol sequence
    for i in range(1, n):
        binary = bin(i)[2:].zfill(len(m))  # Convert index to binary
        x = np.zeros(2)
        for j in range(len(binary)):
            if binary[j] == '1':
                x += direction_numbers[j]
        sobol_seq[i] = x % 1  # Keep values within [0,1]
    
    return sobol_seq

# Generate and plot Sobol sequence
n_points = 50
sobol_points = sobol_2d(n_points)

plt.figure(figsize=(6, 6))
plt.scatter(sobol_points[:, 0], sobol_points[:, 1], c='blue', s=10)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('2D Sobol Sequence (First 50 Points)')
plt.grid(True)
plt.show()