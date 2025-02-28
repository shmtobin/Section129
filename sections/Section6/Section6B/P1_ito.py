#  a) Conversion formula: Ito lemma and Stranovich formulation

"""
Conversion between Itô and Stratonovich Stochastic Integrals

This script demonstrates the conversion between Itô and Stratonovich formulations
of stochastic integrals. The key idea is the midpoint evaluation in Stratonovich
calculus, which introduces a correction term due to the quadratic variation of
the Wiener process.

"""

import numpy as np

def ito_to_stratonovich(mu_ito, sigma, dX_sigma, x, t):
    """
    Convert an Itô SDE to a Stratonovich SDE.

    Parameters:
        mu_ito (function): Drift term in the Itô formulation.
        sigma (function): Diffusion term.
        dX_sigma (function): Derivative of the diffusion term w.r.t. the state variable.
        x (float): State variable.
        t (float): Time variable.

    Returns:
        mu_strat (float): Drift term in the Stratonovich formulation.
    """
    # Evaluate functions at (x, t)
    sigma_val = sigma(x, t)
    dX_sigma_val = dX_sigma(x, t)
    
    # Correction term: -0.5 * sigma * dX_sigma
    correction = -0.5 * sigma_val * dX_sigma_val
    mu_strat = mu_ito(x, t) + correction
    return mu_strat


def stratonovich_to_ito(mu_strat, sigma, dX_sigma, x, t):
    """
    Convert a Stratonovich SDE to an Itô SDE.

    Parameters:
        mu_strat (function): Drift term in the Stratonovich formulation.
        sigma (function): Diffusion term.
        dX_sigma (function): Derivative of the diffusion term w.r.t. the state variable.
        x (float): State variable.
        t (float): Time variable.

    Returns:
        mu_ito (float): Drift term in the Itô formulation.
    """
    # Evaluate functions at (x, t)
    sigma_val = sigma(x, t)
    dX_sigma_val = dX_sigma(x, t)
    
    # Correction term: 0.5 * sigma * dX_sigma
    correction = 0.5 * sigma_val * dX_sigma_val
    mu_ito = mu_strat(x, t) + correction
    return mu_ito


# Example usage
if __name__ == "__main__":
    # Define drift and diffusion terms (example functions)
    def mu_ito(x, t):
        return -x  # Example: Ornstein-Uhlenbeck process

    def sigma(x, t):
        return 1.0  # Constant diffusion

    def dX_sigma(x, t):
        return 0.0  # Derivative of sigma w.r.t. x (0 for constant sigma)

    x_val, t_val = 1.0, 0.0  # Example input values

    # Convert Itô to Stratonovich
    mu_strat = ito_to_stratonovich(mu_ito, sigma, dX_sigma, x_val, t_val)
    print(f"Stratonovich drift term: {mu_strat}")

    # Convert Stratonovich to Itô
    def mu_strat_func(x, t):
        return ito_to_stratonovich(mu_ito, sigma, dX_sigma, x, t)

    mu_ito_converted = stratonovich_to_ito(mu_strat_func, sigma, dX_sigma, x_val, t_val)
    print(f"Converted Itô drift term: {mu_ito_converted}")


# b) Geometric Brownian motion

import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm(mu, sigma, X0, T, N):
    """
    Simulate Geometric Brownian Motion (GBM) under both Itô and Stratonovich interpretations.

    Parameters:
        mu (float): Drift coefficient.
        sigma (float): Diffusion coefficient.
        X0 (float): Initial value of the process.
        T (float): Total simulation time.
        N (int): Number of time steps.

    Returns:
        t (np.ndarray): Time grid.
        X_ito (np.ndarray): Itô-integrated GBM.
        X_strat_analytical (np.ndarray): Analytical Stratonovich solution (given in the problem).
        W (np.ndarray): Wiener process path.
    """
    dt = T / N  # Time step
    t = np.linspace(0, T, N+1)  # Time grid

    # Generate Wiener process increments (dW ~ N(0, dt))
    dW = np.sqrt(dt) * np.random.normal(0, 1, N)
    W = np.cumsum(dW)
    W = np.insert(W, 0, 0)  # W[0] = 0

    # Analytical Stratonovich solution (given in the problem)
    X_strat_analytical = X0 * np.exp(mu * t + sigma * W)

    # Numerical simulation of Itô SDE: dX = (mu + 0.5*sigma²)X dt + sigma X dW
    X_ito = np.zeros(N+1)
    X_ito[0] = X0
    for i in range(N):
        drift = (mu + 0.5 * sigma**2) * X_ito[i]
        diffusion = sigma * X_ito[i]
        X_ito[i+1] = X_ito[i] + drift * dt + diffusion * dW[i]

    return t, X_ito, X_strat_analytical, W

# Parameters
mu = 0.1       # Drift coefficient
sigma = 0.2    # Diffusion coefficient
X0 = 1.0       # Initial value
T = 1.0        # Total time
N = 1000       # Number of time steps

# Simulate paths
t, X_ito, X_strat, W = simulate_gbm(mu, sigma, X0, T, N)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, X_strat, label="Stratonovich (Analytical)", color="blue", linestyle="--")
plt.plot(t, X_ito, label="Itô (Numerical)", color="red", alpha=0.7)
plt.xlabel("Time (t)")
plt.ylabel("X_t")
plt.title("Geometric Brownian Motion: Itô vs Stratonovich")
plt.legend()
plt.grid(True)
plt.show()

# c) Ito differential form

import numpy as np
import matplotlib.pyplot as plt

def simulate_ito_integral(mu, sigma, X0, T, N):
    """
    Simulate the Itô integral x^I(t) = ∫₀ᵗ dX_t^I for GBM.

    Parameters:
        mu (float): Drift coefficient.
        sigma (float): Diffusion coefficient.
        X0 (float): Initial value.
        T (float): Total simulation time.
        N (int): Number of time steps.

    Returns:
        t (np.ndarray): Time grid.
        X_ito (np.ndarray): Simulated trajectory of X_t^I.
        dX_ito (np.ndarray): Increments dX_t^I.
    """
    dt = T / N  # Time step
    t = np.linspace(0, T, N+1)  # Time grid

    # Generate Wiener process increments
    dW = np.sqrt(dt) * np.random.normal(0, 1, N)
    W = np.cumsum(dW)
    W = np.insert(W, 0, 0)  # W[0] = 0

    # Initialize arrays
    X_ito = np.zeros(N+1)
    X_ito[0] = X0
    dX_ito = np.zeros(N)

    # Euler-Maruyama simulation
    for i in range(N):
        drift = (mu + 0.5 * sigma**2) * X_ito[i]
        diffusion = sigma * X_ito[i]
        dX_ito[i] = drift * dt + diffusion * dW[i]
        X_ito[i+1] = X_ito[i] + dX_ito[i]

    return t, X_ito, dX_ito

# Parameters
mu = 0.1    # Drift coefficient
sigma = 0.2 # Diffusion coefficient
X0 = 1.0    # Initial value
T = 10.0    # Total time
N = 100     # Number of time steps

# Simulate trajectory
t, X_ito, dX_ito = simulate_ito_integral(mu, sigma, X0, T, N)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, X_ito, label=r"$X_t^I$ (Itô GBM)", color="blue")
plt.xlabel("Time (t)")
plt.ylabel(r"$X_t^I$")
plt.title(f"Itô Integral Trajectory: $dX_t^I = \left({mu} + \\frac{{({sigma})^2}}{{2}}\\right)X_t^I dt + {sigma} X_t^I dW_t$")
plt.legend()
plt.grid(True)
plt.show()

# d) 