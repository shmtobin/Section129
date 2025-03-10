# ----------------------------
# Part A
# ----------------------------

import scipy.integrate as integrate
import numpy as np

def period(a):
    # Define the potential V(x) = x^4
    def V(x):
        return x**4
    
    # Define the integrand of the formula for the period
    def integrand(x):
        return np.sqrt(V(a) - V(x))

    # Perform the integration
    integral, error = integrate.quad(integrand, 0, a)
    
    # Calculate the period
    T = np.sqrt(8 * 1 * integral)  # m = 1
    return T

# Test the function with a = 2
amplitude = 2
T = period(amplitude)
print(f"Time period for a = {amplitude}: {T}")


# ----------------------------
# Part B
# ----------------------------

def period_fixed_quad(a, N):
    # Define the potential V(x) = x^4
    def V(x):
        return x**4
    
    # Define the integrand of the formula for the period
    def integrand(x):
        return np.sqrt(V(a) - V(x))

    # Perform the fixed quad integration with N points
    integral, error = integrate.fixed_quad(integrand, 0, a, n=N)
    
    # Calculate the period
    T = np.sqrt(8 * 1 * integral)  # m = 1
    return T

# Set the amplitude a = 2
amplitude = 2

# Calculate the period for different values of N
N_values = [10, 20, 40, 80]  # Different values for N
periods = []

for N in N_values:
    T = period_fixed_quad(amplitude, N)
    periods.append(T)
    print(f"Period with N={N}: {T}")

# Estimate the error in the integral by comparing periods for N and 2N
errors = [abs(periods[i] - periods[i-1]) for i in range(1, len(periods))]
print(f"Errors between N and 2N: {errors}")

# Find the value of N where the absolute error is less than 1e-4
for N, error in zip(N_values[1:], errors):
    if error < 1e-4:
        print(f"At N={N}, the absolute error is less than 1e-4.")
        break

# ----------------------------
# Part C
# ----------------------------

def period_quad(a):
    # Define the potential V(x) = x^4
    def V(x):
        return x**4
    
    # Define the integrand of the formula for the period
    def integrand(x):
        return np.sqrt(V(a) - V(x))

    # Perform the quad integration, which also returns an error estimate
    integral, error = integrate.quad(integrand, 0, a)
    
    # Calculate the period
    T = np.sqrt(8 * 1 * integral)  # m = 1
    return T, error

# Calculate the period for a = 2 using quad
amplitude = 2
T_quad, error_quad = period_quad(amplitude)
print(f"Period using quad for a = {amplitude}: {T_quad}")
print(f"Error estimate using quad: {error_quad}")


# ----------------------------
# Part D
# ----------------------------

def period_romberg(a):
    # Define the potential V(x) = x^4
    def V(x):
        return x**4
    
    # Define the integrand of the formula for the period
    def integrand(x):
        return np.sqrt(V(a) - V(x))

    # Perform the Romberg integration
    try:
        integral = integrate.romberg(integrand, 0, a, divmax=10)
        T = np.sqrt(8 * 1 * integral)  # m = 1
    except Exception as e:
        print(f"Error with Romberg integration: {e}")
        return None

    return T

# Calculate the period using Romberg for a = 2
T_romberg = period_romberg(amplitude)
if T_romberg is not None:
    print(f"Period using Romberg for a = {amplitude}: {T_romberg}")

# ----------------------------
# Part E
# ----------------------------

def period_with_error(a, divmax=10, show=False):
    # Define the potential V(x) = x^4
    def V(x):
        return x**4
    
    # Define the integrand of the formula for the period
    def integrand(x):
        return np.sqrt(V(a) - V(x))

    # Perform the Romberg integration with error estimate
    result = integrate.romberg(integrand, 0, a, divmax=divmax, show=show)
    T = np.sqrt(8 * 1 * result)  # m = 1
    return T

# Set amplitude a = 2 and show=True to output detailed results
amplitude = 2
T_with_error = period_with_error(amplitude, divmax=10, show=True)
print(f"Period with error estimate for a = {amplitude}: {T_with_error}")


# ----------------------------
# Part F
# ----------------------------

def period_with_divmax(a, divmax=10):
    # Define the potential V(x) = x^4
    def V(x):
        return x**4
    
    # Define the integrand of the formula for the period
    def integrand(x):
        return np.sqrt(V(a) - V(x))

    # Perform the Romberg integration
    result = integrate.romberg(integrand, 0, a, divmax=divmax)
    T = np.sqrt(8 * 1 * result)  # m = 1
    return T

# Set amplitude a = 2
amplitude = 2

# Calculate the period for different values of divmax
divmax_values = [10, 15]  # Different values for divmax
periods_divmax = []

for divmax in divmax_values:
    T = period_with_divmax(amplitude, divmax)
    periods_divmax.append(T)
    print(f"Period with divmax={divmax}: {T}")

# Compare the periods for divmax=10 and divmax=15
print(f"Change in period when increasing divmax from 10 to 15: {abs(periods_divmax[1] - periods_divmax[0])}")

# ----------------------------
# Part G
# ----------------------------

import matplotlib.pyplot as plt

def period_amplitude(a):
    # Define the potential V(x) = x^4
    def V(x):
        return x**4
    
    # Define the integrand of the formula for the period
    def integrand(x):
        return np.sqrt(V(a) - V(x))

    # Perform the quad integration
    integral, error = integrate.quad(integrand, 0, a)
    
    # Calculate the period
    T = np.sqrt(8 * 1 * integral)  # m = 1
    return T

# Generate amplitudes ranging from a=0 to a=2
amplitudes = np.linspace(0, 2, 100)  # 100 points from 0 to 2
periods = [period_amplitude(a) for a in amplitudes]

# Plot the period as a function of amplitude
plt.plot(amplitudes, periods)
plt.xlabel("Amplitude (a)")
plt.ylabel("Period (T)")
plt.title("Period vs Amplitude for Harmonic Oscillator")
plt.grid(True)
plt.show()