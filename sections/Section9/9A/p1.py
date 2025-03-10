# 9A P1

# ---------------------------- 
# Part a
# ----------------------------

import numpy as np
from scipy.integrate import quad

# Constants
k = 1.38064852e-23  # J/K
h = 6.626e-34  # J·s
pi = np.pi
c = 3e8  # m/s
h_bar = h / (2 * pi)  # ℏ = h / (2π)

# Define the integral integrand using the change of variables (x -> z)
def integrand(z):
    x = z / (1 - z)
    return (x**3) / (np.exp(x) - 1)

# Define the function to compute the integral using a finite range (0 to 1)
def integral():
    result, _ = quad(integrand, 0, 1)
    return result

# Compute the result of the integral
integral_result = integral()
print("Integral result:", integral_result)

# Define the prefactor
prefactor = (k**4 * c**2 * h_bar**4) / (34 * pi**2)

# Total radiated energy per unit area
W = prefactor * integral_result
print("Energy radiated per unit area:", W)

# ---------------------------- 
# Part b
# ----------------------------

# Stefan-Boltzmann constant, using the integral result
def stefan_boltzmann_constant():
    return W / (integral_result)  # since W = σT^4, with T=1

# Calculate the Stefan-Boltzmann constant
sigma = stefan_boltzmann_constant()
print("Stefan-Boltzmann constant:", sigma)

# ---------------------------- 
# Part c
# ----------------------------

# Define the integrand for the infinite range
def integrand_infinite(x):
    return (x**3) / (np.exp(x) - 1)

# Define a function to calculate the integral from 0 to ∞ using quad (supports infinite range)
def integral_infinite():
    result, _ = quad(integrand_infinite, 0, np.inf)
    return result

# Compute the integral from 0 to ∞
integral_infinite_result = integral_infinite()
print("Integral result from 0 to ∞:", integral_infinite_result)

# Calculate the energy radiated per unit area using the result from the infinite range integral
W_infinite = prefactor * integral_infinite_result
print("Energy radiated per unit area (infinite range):", W_infinite)

# Compare the results from the finite range (0 to 1) and infinite range integrals
print("Comparison of the two integrals:")
print("Finite range integral result:", integral_result)
print("Infinite range integral result:", integral_infinite_result)