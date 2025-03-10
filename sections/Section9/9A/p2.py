# ----------------------------
# Part a
# ----------------------------

import numpy as np
import matplotlib.pyplot as plt

# Constants
e = 0.6  # Eccentricity
Tf = 200  # Final time
steps = 100000  # Number of steps
dt = Tf / steps  # Time step

# Initial conditions
q1_0 = 1 - e
q2_0 = 0
q1_dot_0 = 0
q2_dot_0 = np.sqrt((1 + e) / (1 - e))

# Initial position and velocity
q = np.array([q1_0, q2_0])  # Position vector
p = np.array([q1_dot_0, q2_dot_0])  # Velocity vector

# Function to compute the acceleration
def acceleration(q):
    q1, q2 = q
    r_squared = q1**2 + q2**2
    factor = r_squared**(-3/2)
    return np.array([-q1 * factor, -q2 * factor])

# Time integration using Euler method
positions = []
for _ in range(steps):
    positions.append(q)
    # Update position
    q = q + dt * p
    # Update velocity
    p = p + dt * acceleration(q)

# Convert positions to a NumPy array for plotting
positions = np.array(positions)

# Plot the orbit
plt.figure(figsize=(6,6))
plt.plot(positions[:, 0], positions[:, 1], label="Orbit")
plt.xlabel('q1 (x-coordinate)')
plt.ylabel('q2 (y-coordinate)')
plt.title(f'Orbit of the planet (Euler method, e = {e})')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()

# ----------------------------
# Part b
# ----------------------------

import numpy as np
import matplotlib.pyplot as plt

# Constants
e = 0.6  # Eccentricity
Tf = 200  # Final time
steps = 400000  # Number of steps for more precision
dt = Tf / steps  # Time step

# Initial conditions
q1_0 = 1 - e
q2_0 = 0
q1_dot_0 = 0
q2_dot_0 = np.sqrt((1 + e) / (1 - e))

# Initial position and velocity
q = np.array([q1_0, q2_0])  # Position vector
p = np.array([q1_dot_0, q2_dot_0])  # Velocity vector

# Function to compute the Hamiltonian partial derivatives (Hp and Hq)
def Hq(q):
    q1, q2 = q
    r_squared = q1**2 + q2**2
    factor = r_squared**(-3/2)
    return np.array([q1 * factor, q2 * factor])

def Hp(p):
    return p

# Time integration using Symplectic Euler method
positions = []
for _ in range(steps):
    positions.append(q)
    # Update velocity (pn+1 = pn - dt * Hq(pn+1, qn))
    p = p - dt * Hq(q)
    # Update position (qn+1 = qn + dt * Hp(pn+1, qn))
    q = q + dt * Hp(p)

# Convert positions to a NumPy array for plotting
positions = np.array(positions) 

# Plot the orbit
plt.figure(figsize=(6,6))
plt.plot(positions[:, 0], positions[:, 1], label="Orbit")
plt.xlabel('q1 (x-coordinate)')
plt.ylabel('q2 (y-coordinate)')
plt.title(f'Orbit of the planet (Symplectic Euler method, e = {e})')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()