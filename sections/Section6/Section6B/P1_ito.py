"""
Conversion between Itô and Stratonovich Stochastic Integrals

This script demonstrates the conversion between Itô and Stratonovich formulations
of stochastic integrals. The key idea is the midpoint evaluation in Stratonovich
calculus, which introduces a correction term due to the quadratic variation of
the Wiener process.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Part (a): Conversion formula: Ito lemma and Stratonovich formulation
# ----------------------------

def ito_to_stratonovich(mu_ito, sigma, dX_sigma, x, t):
    """
    Convert an Itô SDE to a Stratonovich SDE.
    """
    sigma_val = sigma(x, t)
    dX_sigma_val = dX_sigma(x, t)
    correction = -0.5 * sigma_val * dX_sigma_val
    mu_strat = mu_ito(x, t) + correction
    return mu_strat

def stratonovich_to_ito(mu_strat, sigma, dX_sigma, x, t):
    """
    Convert a Stratonovich SDE to an Itô SDE.
    """
    sigma_val = sigma(x, t)
    dX_sigma_val = dX_sigma(x, t)
    correction = 0.5 * sigma_val * dX_sigma_val
    mu_ito = mu_strat(x, t) + correction
    return mu_ito

# Example usage for conversion formulas (Part a)
if __name__ == "__main__":
    def mu_ito(x, t):
        return -x  # Example: Ornstein-Uhlenbeck process

    def sigma(x, t):
        return 1.0  # Constant diffusion

    def dX_sigma(x, t):
        return 0.0  # Derivative of sigma w.r.t. x

    x_val, t_val = 1.0, 0.0  # Example input values

    mu_strat = ito_to_stratonovich(mu_ito, sigma, dX_sigma, x_val, t_val)
    print(f"Stratonovich drift term: {mu_strat}")

    def mu_strat_func(x, t):
        return ito_to_stratonovich(mu_ito, sigma, dX_sigma, x, t)

    mu_ito_converted = stratonovich_to_ito(mu_strat_func, sigma, dX_sigma, x_val, t_val)
    print(f"Converted Itô drift term: {mu_ito_converted}")

# ----------------------------
# Part (b): Geometric Brownian motion
# ----------------------------
def simulate_gbm(mu, sigma, X0, T, N):
    """
    Simulate Geometric Brownian Motion (GBM) under both Itô and Stratonovich interpretations.
    """
    dt = T / N  
    t = np.linspace(0, T, N+1)

    dW = np.sqrt(dt) * np.random.normal(0, 1, N)
    W = np.cumsum(dW)
    W = np.insert(W, 0, 0)

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

# Parameters for GBM simulation
mu = 0.1       
sigma = 0.2    
X0 = 1.0       
T = 1.0        
N = 1000       

# Simulate paths for Part (b)
t, X_ito, X_strat, W = simulate_gbm(mu, sigma, X0, T, N)

plt.figure(figsize=(10, 6))
plt.plot(t, X_strat, label="Stratonovich (Analytical)", color="blue", linestyle="--")
plt.plot(t, X_ito, label="Itô (Numerical)", color="red", alpha=0.7)
plt.xlabel("Time (t)")
plt.ylabel("X_t")
plt.title("Geometric Brownian Motion: Itô vs Stratonovich")
plt.legend()
plt.grid(True)
plt.savefig("plots/gbm_ito_vs_stratonovich.png", dpi=300)  # Saves plot for Part (b)
plt.show()

# ----------------------------
# Part (c): Ito differential form
# ----------------------------
def simulate_ito_integral(mu, sigma, X0, T, N):
    """
    Simulate the Itô integral x^I(t) = ∫₀ᵗ dX_t^I for GBM.
    """
    dt = T / N  
    t = np.linspace(0, T, N+1)

    dW = np.sqrt(dt) * np.random.normal(0, 1, N)
    W = np.cumsum(dW)
    W = np.insert(W, 0, 0)

    X_ito = np.zeros(N+1)
    X_ito[0] = X0
    dX_ito = np.zeros(N)

    for i in range(N):
        drift = (mu + 0.5 * sigma**2) * X_ito[i]
        diffusion = sigma * X_ito[i]
        dX_ito[i] = drift * dt + diffusion * dW[i]
        X_ito[i+1] = X_ito[i] + dX_ito[i]

    return t, X_ito, dX_ito

# Parameters for Itô differential form
mu = 0.1    
sigma = 0.2 
X0 = 1.0    
T = 10.0    
N = 100     

t, X_ito, dX_ito = simulate_ito_integral(mu, sigma, X0, T, N)

plt.figure(figsize=(10, 6))
plt.plot(t, X_ito, label=r"$X_t^I$ (Itô GBM)", color="blue")
plt.xlabel("Time (t)")
plt.ylabel(r"$X_t^I$")
plt.title(f"Itô Integral Trajectory: $dX_t^I = \\left({mu} + \\frac{{({sigma})^2}}{{2}}\\right)X_t^I dt + {sigma} X_t^I dW_t$")
plt.legend()
plt.grid(True)
plt.savefig("plots/ito_integral_trajectory.png", dpi=300)  # Saves plot for Part (c)
plt.show()

# ----------------------------
# Part (d): Stratonovich Differential Form
# ----------------------------
def simulate_stratonovich_integral(mu, sigma, X0, T, N, dW):
    """
    Simulate a single trajectory of GBM using the Stratonovich integrator.
    """
    dt = T / N
    t = np.linspace(0, T, N+1)
    X = np.zeros(N+1)
    X[0] = X0
    
    for i in range(N):
        X_predictor = X[i] + mu * X[i] * dt + sigma * X[i] * dW[i]
        drift_avg = 0.5 * (mu * X[i] + mu * X_predictor)
        diffusion_avg = 0.5 * (sigma * X[i] + sigma * X_predictor)
        X[i+1] = X[i] + drift_avg * dt + diffusion_avg * dW[i]
    return t, X

# For a single trajectory, using N = 100 as specified
N_single = 100
dt_single = T / N_single
np.random.seed(12345)
dW_single = np.sqrt(dt_single) * np.random.normal(0, 1, size=N_single)
t_single, X_strat_single = simulate_stratonovich_integral(mu, sigma, X0, T, N_single, dW_single)

plt.figure(figsize=(8, 5))
plt.plot(t_single, X_strat_single, label="Stratonovich Trajectory", color='purple')
plt.xlabel("Time t")
plt.ylabel("X(t)")
plt.title("Single Stratonovich Trajectory (N = 100)")
plt.legend()
plt.grid(True)
plt.savefig("plots/single_stratonovich_trajectory.png", dpi=300)  # Saves plot for Part (d)
plt.show()

# ----------------------------
# Part (e): Statistics vs. N (Time Resolution)
# ----------------------------
def simulate_ito_ensemble(mu, sigma, X0, T, N, M, dW_ensemble):
    """
    Simulate an ensemble of M trajectories for the Ito integrator.
    """
    dt = T / N
    X = np.zeros((M, N+1))
    X[:, 0] = X0
    for j in range(N):
        X[:, j+1] = X[:, j] + ((mu + 0.5 * sigma**2) * X[:, j]) * dt + sigma * X[:, j] * dW_ensemble[:, j]
    return X

def simulate_stratonovich_ensemble(mu, sigma, X0, T, N, M, dW_ensemble):
    """
    Simulate an ensemble of M trajectories for the Stratonovich integrator using the Heun method.
    """
    dt = T / N
    X = np.zeros((M, N+1))
    X[:, 0] = X0
    for j in range(N):
        X_predictor = X[:, j] + mu * X[:, j] * dt + sigma * X[:, j] * dW_ensemble[:, j]
        drift_avg = 0.5 * (mu * X[:, j] + mu * X_predictor)
        diffusion_avg = 0.5 * (sigma * X[:, j] + sigma * X_predictor)
        X[:, j+1] = X[:, j] + drift_avg * dt + diffusion_avg * dW_ensemble[:, j]
    return X

# Parameters for ensemble statistics
T = 10.0
X0 = 1.0
mu = 0.1
sigma = 0.2
M = 1000
N_values = np.logspace(1, 4, num=10, dtype=int)

ito_means = []
ito_vars = []
strat_means = []
strat_vars = []

for N in N_values:
    dt = T / N
    np.random.seed(12345)
    dW_ensemble = np.sqrt(dt) * np.random.normal(0, 1, size=(M, N))
    
    X_ito = simulate_ito_ensemble(mu, sigma, X0, T, N, M, dW_ensemble)
    X_strat = simulate_stratonovich_ensemble(mu, sigma, X0, T, N, M, dW_ensemble)
    
    ito_means.append(np.mean(X_ito[:, -1]))
    ito_vars.append(np.var(X_ito[:, -1]))
    strat_means.append(np.mean(X_strat[:, -1]))
    strat_vars.append(np.var(X_strat[:, -1]))

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs[0, 0].plot(N_values, ito_means, marker='o')
axs[0, 0].set_xscale('log')
axs[0, 0].set_title("Ito Integrator: Mean of X(T)")
axs[0, 0].set_xlabel("N (time steps, log scale)")
axs[0, 0].set_ylabel("Mean")

axs[0, 1].plot(N_values, ito_vars, marker='o', color='red')
axs[0, 1].set_xscale('log')
axs[0, 1].set_title("Ito Integrator: Variance of X(T)")
axs[0, 1].set_xlabel("N (time steps, log scale)")
axs[0, 1].set_ylabel("Variance")

axs[1, 0].plot(N_values, strat_means, marker='o', color='green')
axs[1, 0].set_xscale('log')
axs[1, 0].set_title("Stratonovich Integrator: Mean of X(T)")
axs[1, 0].set_xlabel("N (time steps, log scale)")
axs[1, 0].set_ylabel("Mean")

axs[1, 1].plot(N_values, strat_vars, marker='o', color='orange')
axs[1, 1].set_xscale('log')
axs[1, 1].set_title("Stratonovich Integrator: Variance of X(T)")
axs[1, 1].set_xlabel("N (time steps, log scale)")
axs[1, 1].set_ylabel("Variance")

plt.tight_layout()
plt.savefig("plots/statistics_vs_N.png", dpi=300)  # Saves plot for Part (e)
plt.show()

# ----------------------------
# Part (f): Functional Dynamics on GBM
# ----------------------------
def simulate_functional_ito_ensemble(mu, sigma, X0, T, N, M, seed=None):
    """
    Simulate the functional stopping dynamics using the Ito integrator.
    F_I(t) = ∫₀ᵗ X(s)^2 dX_I(s) with left-point evaluation.
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    t = np.linspace(0, T, N+1)
    X = np.zeros((M, N+1))
    F = np.zeros((M, N+1))
    X[:, 0] = X0
    for j in range(N):
        dW = np.sqrt(dt) * np.random.normal(0, 1, size=M)
        dX = ((mu + 0.5 * sigma**2) * X[:, j]) * dt + sigma * X[:, j] * dW
        F[:, j+1] = F[:, j] + (X[:, j]**2) * dX
        X[:, j+1] = X[:, j] + dX
    return t, X, F

def simulate_functional_strat_ensemble(mu, sigma, X0, T, N, M, seed=None):
    """
    Simulate the functional stopping dynamics using the Stratonovich integrator.
    F_S(t) = ∫₀ᵗ [0.5*(X(s)^2 + X(s+ds)^2)] dX_S(s) with midpoint evaluation.
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    t = np.linspace(0, T, N+1)
    X = np.zeros((M, N+1))
    F = np.zeros((M, N+1))
    X[:, 0] = X0
    for j in range(N):
        dW = np.sqrt(dt) * np.random.normal(0, 1, size=M)
        X_predictor = X[:, j] + mu * X[:, j] * dt + sigma * X[:, j] * dW
        drift_avg = 0.5 * (mu * X[:, j] + mu * X_predictor)
        diffusion_avg = 0.5 * (sigma * X[:, j] + sigma * X_predictor)
        dX = drift_avg * dt + diffusion_avg * dW
        X_next = X[:, j] + dX
        f_mid = 0.5 * (X[:, j]**2 + X_next**2)
        F[:, j+1] = F[:, j] + f_mid * dX
        X[:, j+1] = X_next
    return t, X, F

# Ensemble analysis for functional dynamics (Part f)
T_func = 10.0
X0 = 1.0
mu = 0.1
sigma = 0.2
M = 1000
N_values = np.logspace(1, 4, num=10, dtype=int)

ito_F_means = []
ito_F_vars = []
strat_F_means = []
strat_F_vars = []

for N in N_values:
    seed_val = 12345
    t, X_ito_func, F_ito = simulate_functional_ito_ensemble(mu, sigma, X0, T_func, N, M, seed=seed_val)
    t, X_strat_func, F_strat = simulate_functional_strat_ensemble(mu, sigma, X0, T_func, N, M, seed=seed_val)
    ito_F_means.append(np.mean(F_ito[:, -1]))
    ito_F_vars.append(np.var(F_ito[:, -1]))
    strat_F_means.append(np.mean(F_strat[:, -1]))
    strat_F_vars.append(np.var(F_strat[:, -1]))

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs[0, 0].plot(N_values, ito_F_means, marker='o')
axs[0, 0].set_xscale('log')
axs[0, 0].set_title("Ito Functional: Mean of F(T)")
axs[0, 0].set_xlabel("N (time steps, log scale)")
axs[0, 0].set_ylabel("Mean")

axs[0, 1].plot(N_values, ito_F_vars, marker='o', color='red')
axs[0, 1].set_xscale('log')
axs[0, 1].set_title("Ito Functional: Variance of F(T)")
axs[0, 1].set_xlabel("N (time steps, log scale)")
axs[0, 1].set_ylabel("Variance")

axs[1, 0].plot(N_values, strat_F_means, marker='o', color='green')
axs[1, 0].set_xscale('log')
axs[1, 0].set_title("Stratonovich Functional: Mean of F(T)")
axs[1, 0].set_xlabel("N (time steps, log scale)")
axs[1, 0].set_ylabel("Mean")

axs[1, 1].plot(N_values, strat_F_vars, marker='o', color='orange')
axs[1, 1].set_xscale('log')
axs[1, 1].set_title("Stratonovich Functional: Variance of F(T)")
axs[1, 1].set_xlabel("N (time steps, log scale)")
axs[1, 1].set_ylabel("Variance")

plt.tight_layout()
plt.savefig("plots/functional_statistics_vs_N.png", dpi=300)  # Saves plot for Part (f)
plt.show()

# ----------------------------
# Part (g): Autocorrelation of the Stopping Function F(t)
# ----------------------------
def simulate_functional_ito_fixed_dt(mu, sigma, X0, T, dt, M, seed=None):
    """
    Simulate an ensemble for F(t) using the Ito integrator with a fixed dt.
    """
    if seed is not None:
        np.random.seed(seed)
    N = int(T / dt)
    t = np.linspace(0, T, N+1)
    X = np.zeros((M, N+1))
    F = np.zeros((M, N+1))
    X[:, 0] = X0
    for j in range(N):
        dW = np.sqrt(dt) * np.random.normal(0, 1, size=M)
        dX = ((mu + 0.5 * sigma**2) * X[:, j]) * dt + sigma * X[:, j] * dW
        F[:, j+1] = F[:, j] + (X[:, j]**2) * dX
        X[:, j+1] = X[:, j] + dX
    return t, X, F

def compute_autocorrelation(F, dt, ref_index, max_lag):
    """
    Compute the autocorrelation function for F(t) at a given reference index.
    """
    lag_steps = int(max_lag / dt)
    C = np.zeros(lag_steps + 1)
    for lag in range(lag_steps + 1):
        C[lag] = np.mean(F[:, ref_index] * F[:, ref_index + lag])
    tau = np.arange(lag_steps + 1) * dt
    return tau, C

# Parameters for autocorrelation analysis
T_auto = 40.0
dt_auto = 0.1
M_auto = 1000
t_auto, X_auto, F_auto = simulate_functional_ito_fixed_dt(mu, sigma, X0, T_auto, dt_auto, M_auto, seed=54321)

ref_times = [5, 10, 20, 30]
ref_indices = [int(t_ref / dt_auto) for t_ref in ref_times]
max_lag = 10.0

plt.figure(figsize=(10, 6))
for rt, idx in zip(ref_times, ref_indices):
    tau, C = compute_autocorrelation(F_auto, dt_auto, idx, max_lag)
    plt.plot(tau, C, label=f"Reference t = {rt}")

plt.xlabel("Lag τ")
plt.ylabel("Autocorrelation C(τ)")
plt.title("Autocorrelation of the Stopping Function F(t)")
plt.legend()
plt.grid(True)
plt.savefig("plots/autocorrelation_stopping_function.png", dpi=300)  # Saves plot for Part (g)
plt.show()