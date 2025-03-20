# ----------------------------
# Part a
# ----------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Define the noisy ϕ4 theory Hamiltonian
def H(theta):
    return theta**4 - 8*theta**2 - 2*np.cos(4 * np.pi * theta)

# Gradient of H with respect to theta
def grad_H(theta):
    return 4*theta**3 - 16*theta - 8*np.pi * np.sin(4 * np.pi * theta)

# Gradient descent update rule
def gradient_descent(theta_init, alpha, num_iterations=100):
    theta = theta_init
    theta_history = []
    
    for _ in range(num_iterations):
        theta_history.append(theta)
        theta = theta - alpha * grad_H(theta)
        
        # Check for convergence: stop if the gradient is close to zero
        if np.linalg.norm(grad_H(theta)) < 1e-6:
            break
    
    return np.array(theta_history)

# Parameters for gradient descent
alpha = 0.05  # Learning rate
num_iterations = 100  # Number of iterations

# Initial guesses for theta
initial_guesses = [-1, 0.5, 3]

# Create a plot for the Hamiltonian
theta_vals = np.linspace(-2, 2, 400)
H_vals = H(theta_vals)

plt.figure(figsize=(8, 6))
plt.plot(theta_vals, H_vals, label="Hamiltonian H(θ)", color='blue')

# For each initial guess, perform gradient descent and plot the steps
for theta_init in initial_guesses:
    theta_history = gradient_descent(theta_init, alpha, num_iterations)
    
    # Plot each step as a red dot on the Hamiltonian plot
    plt.scatter(theta_history, H(theta_history), color='red', label=f"Starting at θ0 = {theta_init}")

# Add labels and a legend
plt.title("Gradient Descent for Noisy ϕ4 Theory in 1D")
plt.xlabel("θ")
plt.ylabel("H(θ)")
plt.legend()

# Save the plot as an image
plt.savefig(f"plots/gradient_descent_plot.png")
plt.show()

# ----------------------------
# Creating a video of the descent steps
# ----------------------------

# Ensure the directory for frames exists
frame_dir = "plots"
os.makedirs(frame_dir, exist_ok=True)

# Initialize the VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'plots/gradient_descent_video.avi', fourcc, 10.0, (640, 480))

# For each initial guess, perform gradient descent and save frames
for theta_init in initial_guesses:
    theta_history = gradient_descent(theta_init, alpha, num_iterations)
    
    for i in range(len(theta_history)):
        plt.figure(figsize=(8, 6))
        plt.plot(theta_vals, H(theta_vals), label="Hamiltonian H(θ)", color='blue')
        plt.scatter(theta_history[:i+1], H(theta_history[:i+1]), color='red')  # Plot steps up to the current one
        plt.title("Gradient Descent for Noisy ϕ4 Theory in 1D")
        plt.xlabel("θ")
        plt.ylabel("H(θ)")
        plt.legend()
        
        # Save the frame
        frame_path = os.path.join(frame_dir, "temp_frame.png")
        plt.savefig(frame_path)
        frame = cv2.imread(frame_path)
        
        if frame is not None:
            out.write(frame)
        else:
            print(f"Warning: Frame at {frame_path} could not be loaded.")
        
        plt.close()

# Release the video writer
out.release()


# ----------------------------
# Part b
# ----------------------------

import numpy as np

# Define the noisy ϕ4 theory Hamiltonian
def H(theta):
    return theta**4 - 8*theta**2 - 2*np.cos(4 * np.pi * theta)

# Metropolis-Hastings algorithm
def metropolis_hastings(theta_init, beta, num_steps=1000, sigma=0.1):
    theta = theta_init
    theta_history = [theta]
    
    for _ in range(num_steps):
        # Propose a new value for theta
        theta_star = theta + np.random.normal(0, sigma)
        
        # Compute the change in the Hamiltonian
        delta_H = H(theta_star) - H(theta)
        
        # Compute the acceptance ratio
        r = np.exp(-beta * delta_H)
        
        # Accept or reject the new value
        if r > 1:
            theta = theta_star  # Always accept if r > 1
        else:
            u = np.random.uniform(0, 1)
            if u < r:
                theta = theta_star  # Accept with probability r
        
        # Record the current value of theta
        theta_history.append(theta)
    
    return np.array(theta_history)

# Parameters for Metropolis-Hastings
beta = 1.0  # Inverse temperature
sigma = 0.1  # Standard deviation for the proposal distribution
num_steps = 1000  # Number of steps in the Markov chain

# Initial guesses for theta
initial_guesses = [-1, 0.5, 3]

# Create a plot for the Hamiltonian
theta_vals = np.linspace(-2, 2, 400)
H_vals = H(theta_vals)

plt.figure(figsize=(8, 6))
plt.plot(theta_vals, H_vals, label="Hamiltonian H(θ)", color='blue')

# For each initial guess, perform Metropolis-Hastings and plot the steps
for theta_init in initial_guesses:
    theta_history = metropolis_hastings(theta_init, beta, num_steps, sigma)
    
    # Plot the theta values over the course of the steps
    plt.scatter(theta_history, H(theta_history), color='red', label=f"Starting at θ0 = {theta_init}")

# Add labels and a legend
plt.title("Metropolis–Hastings Algorithm for Noisy ϕ4 Theory in 1D")
plt.xlabel("θ")
plt.ylabel("H(θ)")
plt.legend()

# Save the plot as an image
plt.savefig(f"plots/metropolis_hastings_plot.png")
plt.show()

# ----------------------------
# Part c
# ----------------------------

# Simulated Annealing Algorithm with Cooling Schedule
def simulated_annealing(theta_init, beta_init, num_steps=1000, sigma=0.1, delta_beta=0.01):
    theta = theta_init
    beta = beta_init
    theta_history = [theta]
    beta_history = [beta]
    
    for _ in range(num_steps):
        # Propose a new value for theta
        theta_star = theta + np.random.normal(0, sigma)
        
        # Compute the change in the Hamiltonian
        delta_H = H(theta_star) - H(theta)
        
        # Compute the acceptance ratio
        r = np.exp(-beta * delta_H)
        
        # Accept or reject the new value
        if r > 1:
            theta = theta_star  # Always accept if r > 1
        else:
            u = np.random.uniform(0, 1)
            if u < r:
                theta = theta_star  # Accept with probability r
        
        # Update the inverse temperature (simulating annealing)
        beta += delta_beta
        
        # Record the current values of theta and beta
        theta_history.append(theta)
        beta_history.append(beta)
    
    return np.array(theta_history), np.array(beta_history)

# Parameters for Simulated Annealing
beta_init = 0.1  # Initial inverse temperature
sigma = 0.1  # Standard deviation for the proposal distribution
delta_beta = 0.001  # Rate of change of the inverse temperature
num_steps = 1000  # Number of steps in the chain

# Initial guesses for theta
initial_guesses = [-1, 0.5, 3]

# Create a plot for the Hamiltonian
theta_vals = np.linspace(-2, 2, 400)
H_vals = H(theta_vals)

plt.figure(figsize=(8, 6))
plt.plot(theta_vals, H_vals, label="Hamiltonian H(θ)", color='blue')

# For each initial guess, perform Simulated Annealing and plot the steps
for theta_init in initial_guesses:
    theta_history, beta_history = simulated_annealing(theta_init, beta_init, num_steps, sigma, delta_beta)
    
    # Plot the theta values over the course of the steps
    plt.scatter(theta_history, H(theta_history), color='red', label=f"Starting at θ0 = {theta_init}")

# Add labels and a legend
plt.title("Simulated Annealing with Cooling Schedule for Noisy ϕ4 Theory in 1D")
plt.xlabel("θ")
plt.ylabel("H(θ)")
plt.legend()

# Save the plot as an image
plt.savefig(f"plots/simulated_annealing_plot.png")
plt.show()