"""
Task 2: Langevin Equation

The Langevin equation under consideration is:
    dv/dt = -γ v(t) + √(2D) η(t),
with initial condition v(0) = v0, and where the noise is given by
    η(t) = dX(t)/dt,  with  X(t) = W(t)^2.
Using the Ito integrator, we update:
    v(t+dt) = v(t) - γ v(t) dt + √(2D) * [X(t+dt) - X(t)].
We simulate an ensemble of trajectories, compute the mean and variance of v(t),
and save all generated plots to the "plots" directory with 300 dpi, with filenames ending with _task2.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the output directory exists
if not os.path.exists("plots"):
    os.makedirs("plots")

# ----------------------------
# Simulation parameters
# ----------------------------
T = 10.0         # Total simulation time
dt = 0.01        # Time step
N = int(T / dt)  # Number of time steps
M = 1000         # Number of ensemble trajectories

gamma = 1.0      # Friction coefficient
D = 1.0          # Diffusion constant
v0 = 0.0         # Initial velocity

# ----------------------------
# Step 1: Simulate the driving process X(t) = W(t)^2
# ----------------------------
# Generate Wiener process increments for each trajectory:
dW = np.sqrt(dt) * np.random.normal(0, 1, size=(M, N))
# Compute Wiener paths by cumulative sum; prepend initial 0:
W = np.concatenate([np.zeros((M, 1)), np.cumsum(dW, axis=1)], axis=1)

# Define X(t) = [W(t)]^2 (elementwise square)
X = W**2

# Compute increments dX = X(t+dt) - X(t)
dX = np.diff(X, axis=1)  # shape: (M, N)

# ----------------------------
# Step 2: Integrate the Langevin equation using the Ito scheme
# ----------------------------
# The SDE for v(t) is:
#   dv = -gamma*v(t)*dt + sqrt(2D)*dX
# Initialize velocity array:
v = np.zeros((M, N+1))
v[:, 0] = v0

# Time integration:
for i in range(N):
    v[:, i+1] = v[:, i] + (-gamma * v[:, i] * dt + np.sqrt(2 * D) * dX[:, i])

# Create time vector:
time = np.linspace(0, T, N+1)

# ----------------------------
# Step 3: Compute ensemble statistics (mean and variance)
# ----------------------------
mean_v = np.mean(v, axis=0)
var_v = np.var(v, axis=0)

# ----------------------------
# Step 4: Plot and save results
# ----------------------------

# (a) Plot a few sample trajectories
plt.figure(figsize=(10, 6))
# Select 10 random trajectories to plot:
sample_indices = np.random.choice(M, size=10, replace=False)
for idx in sample_indices:
    plt.plot(time, v[idx, :], alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Velocity v(t)")
plt.title("Sample Langevin Trajectories (Task 2)")
plt.grid(True)
plt.savefig("plots/plot_langevin_sample_trajectories_task2.png", dpi=300)
plt.show()

# (b) Plot ensemble mean and variance vs. time
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

axs[0].plot(time, mean_v, color='blue', label='Mean of v(t)')
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Mean Velocity")
axs[0].set_title("Ensemble Mean of Velocity (Task 2)")
axs[0].grid(True)
axs[0].legend()

axs[1].plot(time, var_v, color='red', label='Variance of v(t)')
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Variance")
axs[1].set_title("Ensemble Variance of Velocity (Task 2)")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.savefig("plots/plot_langevin_mean_variance_task2.png", dpi=300)
plt.show()