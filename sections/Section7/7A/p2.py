import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import sqrt, pi, exp

# =============================================================================
# Helper Functions
# =============================================================================

def numerical_hessian(f, x, epsilon=1e-5):
    """
    Compute the Hessian matrix of f at x using central finite differences.
    """
    x = np.atleast_1d(x)
    n = x.size
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_ijp = np.copy(x)
            x_ijp[i] += epsilon
            x_ijp[j] += epsilon
            f_ijp = f(x_ijp)
            
            x_ijm = np.copy(x)
            x_ijm[i] += epsilon
            x_ijm[j] -= epsilon
            f_ijm = f(x_ijm)
            
            x_jim = np.copy(x)
            x_jim[i] -= epsilon
            x_jim[j] += epsilon
            f_jim = f(x_jim)
            
            x_jim2 = np.copy(x)
            x_jim2[i] -= epsilon
            x_jim2[j] -= epsilon
            f_jim2 = f(x_jim2)
            
            hessian[i, j] = (f_ijp - f_ijm - f_jim + f_jim2) / (4 * epsilon**2)
    return hessian

# =============================================================================
# Task 2a: Fitting Decay Data for the Vacuum Dataset
# =============================================================================

def fit_vacuum_decay(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    data = np.array(data)
    N = len(data)
    lambda_hat = np.mean(data - 1)
    fisher_info = N / (lambda_hat**2)
    variance_lambda = lambda_hat**2 / N

    print(f"Vacuum Data: N = {N}, λ̂ = {lambda_hat:.4f}, Variance = {variance_lambda:.4f}, Fisher Info = {fisher_info:.4f}")
    return data, lambda_hat, variance_lambda, fisher_info

def plot_vacuum_fit(data, lambda_hat, bins=50):
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, density=True, alpha=0.5, label='Vacuum Data')
    x_vals = np.linspace(max(1, data.min()), data.max(), 500)
    pdf = (1/lambda_hat) * np.exp(-(x_vals - 1)/lambda_hat)
    plt.plot(x_vals, pdf, 'r-', lw=2, label=f'Exponential Fit (λ = {lambda_hat:.3f})')
    plt.xlabel('Decay Distance x')
    plt.ylabel('Probability Density')
    plt.title('Vacuum Decay Data and Exponential Fit')
    plt.legend()
    plt.show()

# =============================================================================
# Task 2a & 2b: Fitting the Cavity Dataset with a Mixture Model
# =============================================================================

def neg_log_likelihood(params, data):
    f_mix, lam, mu, sigma = params
    if f_mix < 0 or f_mix > 1 or lam <= 0 or sigma <= 0:
        return np.inf
    P_exp = (1/lam) * np.exp(-(data - 1)/lam)
    F_gauss = (1/(sigma * np.sqrt(2*pi))) * np.exp(-0.5 * ((data - mu)/sigma)**2)
    pdf = (1 - f_mix) * P_exp + f_mix * F_gauss
    eps = 1e-10
    log_likelihood = np.sum(np.log(pdf + eps))
    return -log_likelihood

def fit_cavity_decay(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    data = np.array(data)

    f0 = 0.2
    lam0 = np.mean(data) - 1
    mu0 = np.mean(data)
    sigma0 = np.std(data)
    initial_params = [f0, lam0, mu0, sigma0]

    bounds = [(0, 1), (1e-5, None), (None, None), (1e-5, None)]
    
    result = minimize(neg_log_likelihood, initial_params, args=(data,), bounds=bounds)
    if not result.success:
        print("Cavity fit optimization failed:", result.message)
        return None
    best_params = result.x
    best_nll = result.fun

    H = numerical_hessian(lambda params: neg_log_likelihood(params, data), best_params)
    fisher_info_matrix = H
    try:
        cov_matrix = np.linalg.inv(fisher_info_matrix)
    except np.linalg.LinAlgError:
        cov_matrix = None

    print("Cavity Data Fit Results:")
    print(f"Mixing fraction, f    = {best_params[0]:.4f}")
    print(f"Decay constant, λ     = {best_params[1]:.4f}")
    print(f"Gaussian mean, µ      = {best_params[2]:.4f}")
    print(f"Gaussian std, σ       = {best_params[3]:.4f}")
    if cov_matrix is not None:
        print("\nParameter variances (diagonal of covariance matrix):")
        print(f"Var(f)   = {cov_matrix[0,0]:.6f}")
        print(f"Var(λ)   = {cov_matrix[1,1]:.6f}")
        print(f"Var(µ)   = {cov_matrix[2,2]:.6f}")
        print(f"Var(σ)   = {cov_matrix[3,3]:.6f}")
    else:
        print("Covariance matrix could not be computed (singular Hessian).")
    print(f"Best negative log-likelihood: {best_nll:.4f}\n")
    
    return data, best_params, cov_matrix, fisher_info_matrix, best_nll

def plot_cavity_fit(data, best_params, bins=50):
    f_mix, lam, mu, sigma = best_params
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, density=True, alpha=0.5, label='Cavity Data')
    x_vals = np.linspace(data.min(), data.max(), 500)
    P_exp = (1/lam) * np.exp(-(x_vals - 1)/lam)
    F_gauss = (1/(sigma * np.sqrt(2*pi))) * np.exp(-0.5*((x_vals - mu)/sigma)**2)
    mix_pdf = (1 - f_mix) * P_exp + f_mix * F_gauss
    plt.plot(x_vals, mix_pdf, 'r-', lw=2, label='Fitted Mixture Model')
    plt.xlabel('Decay Distance x')
    plt.ylabel('Probability Density')
    plt.title('Cavity Decay Data and Mixture Model Fit')
    plt.legend()
    plt.show()

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    # ----- Vacuum Dataset (Pure Exponential) -----
    vacuum_file = "Vacuum_decay_dataset.json"
    vacuum_data, lambda_hat, var_lambda, fisher_info_vac = fit_vacuum_decay(vacuum_file)
    plot_vacuum_fit(vacuum_data, lambda_hat)
    
    # ----- Cavity Dataset (Mixture: Exponential + Gaussian) -----
    cavity_file = "Cavity_decay_dataset.json"
    cavity_data, best_params, cov_matrix, fisher_info_matrix, best_nll = fit_cavity_decay(cavity_file)
    plot_cavity_fit(cavity_data, best_params)
