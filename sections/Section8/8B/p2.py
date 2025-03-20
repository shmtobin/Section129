import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

# Define the polynomial integral
def polynomial_integral(k, a, b):
    return (b**(k+1) - a**(k+1)) / (k+1)

# Define the Fermi-Dirac integral
def fermi_dirac_integral(k, a, b):
    return (1/k) * (np.log(np.exp(k*b) + 1) - np.log(np.exp(k*a) + 1))

# Quadrature methods
def midpoint_rule(f, a, b, N):
    x = np.linspace(a + (b-a)/(2*N), b - (b-a)/(2*N), N)
    return (b-a)/N * np.sum(f(x))

def trapezoidal_rule(f, a, b, N):
    x = np.linspace(a, b, N+1)
    return (b-a)/(2*N) * (f(x[0]) + 2*np.sum(f(x[1:-1])) + f(x[-1]))

def simpsons_rule(f, a, b, N):
    if N % 2 == 1:
        N += 1  # Simpson's rule requires an even number of intervals
    x = np.linspace(a, b, N+1)
    return (b-a)/(3*N) * (f(x[0]) + 4*np.sum(f(x[1:-1:2])) + 2*np.sum(f(x[2:-2:2])) + f(x[-1]))

def gauss_legendre_rule(f, a, b, N):
    x, w = np.polynomial.legendre.leggauss(N)
    x_mapped = 0.5 * (b-a) * x + 0.5 * (b+a)
    return 0.5 * (b-a) * np.sum(w * f(x_mapped))

# Compute heatmaps
def compute_heatmap(quadrature_method, true_integral, a, b, k_values, N_values, f):
    errors = np.zeros((len(k_values), len(N_values)))
    for i, k in enumerate(k_values):
        I_true = true_integral(k, a, b)
        for j, N in enumerate(N_values):
            I_num = quadrature_method(lambda x: f(x, k), a, b, N)
            errors[i, j] = abs((2*I_true - I_num) / (I_true + I_num))
    return errors

# Function definitions for f(x, k)
def polynomial_function(x, k):
    return x**k

def fermi_dirac_function(x, k):
    return 1 / (1 + np.exp(-k*x))

# Parameters
a, b = 0, 1
k_values = np.arange(0, 11)
N_values = np.logspace(1, 5, 50, dtype=int)
methods = [(midpoint_rule, "Midpoint"), (trapezoidal_rule, "Trapezoidal"), 
           (simpsons_rule, "Simpson"), (gauss_legendre_rule, "Gauss-Legendre")]

# Generate heatmaps
def plot_heatmaps(f, true_integral, title):
    for method, method_name in methods:
        errors = compute_heatmap(method, true_integral, a, b, k_values, N_values, f)
        plt.figure(figsize=(8, 6))
        plt.imshow(np.log10(errors + 1e-16), aspect='auto', origin='lower', 
                   extent=[np.log10(N_values[0]), np.log10(N_values[-1]), k_values[0], k_values[-1]])
        plt.colorbar(label='log10(Relative Error)')
        plt.xlabel("log10(N)")
        plt.ylabel("k")
        plt.title(f"{method_name} - {title}")
        plt.savefig(f'plots/plot_heatmaps.png', dpi=300)
        plt.show()

# ----------------------------
# Part a: Polynomial
# ----------------------------
plot_heatmaps(polynomial_function, polynomial_integral, "Polynomial Integral")

# ----------------------------
# Part b: Fermi-Dirac
# ----------------------------
plot_heatmaps(fermi_dirac_function, fermi_dirac_integral, "Fermi-Dirac Integral")