# c)  Plot both ⟨n_0⟩_C and ⟨n_ϵ⟩_C on the same figure.

import numpy as np
import matplotlib.pyplot as plt

# set Beta*epsilon = x, which will be plotted as independent
x = np.linspace(0, 10, 100)

# Define functions
n0_c = 1 / (1 + np.exp(-x))
n1 = 1 / (1 + np.exp(x))

# Plot
plt.plot(x, n0_c, label=r'$\langle n_0 \rangle_c = \frac{1}{1+e^{-x}}$', color='blue')
plt.plot(x, n1, label=r'$\langle n_1 \rangle = \frac{1}{1+e^x}$', color='red')

# Labels and title
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Plot of $\\langle n_0 \\rangle_c$ and $\\langle n_1 \\rangle$')
plt.legend()
plt.savefig('Plots/P2_c', bbox_inches='tight')
plt.grid()

# Show plot
plt.show()


# low effort stab at completing part i because it seems interesting but is 
# so many parts away
import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Constants
N = 1e5           # Total number of particles
k_B = 1           # Boltzmann constant (set to 1 for simplicity)
delta = 0.01      # Energy spacing between excited states
M = 10000         # Number of excited states
excited_energies = delta * np.arange(1, M + 1)  # Excited state energies

# Temperature range setup
T_min = 0.1
T_max = 5.0
dT = 0.1
temps = np.arange(T_min, T_max + dT, dT)

# Arrays to store results
mu_list = []
n0_list = []
log_n0_list = []
dn0_dT_list = []
U_list = []
Cv_list = []

# Variables to store previous values for derivatives
prev_n0 = None
prev_U = None
prev_T = None

for T in temps:
    # Function to compute the difference between total particles and N
    def equation(mu):
        # Ground state occupation
        if mu >= 0:
            return np.inf  # To avoid mu >= 0
        exponent = -mu / (k_B * T)
        if exponent < 1e-10:
            n0 = -k_B * T / mu  # Approximation for small exponent
        else:
            n0 = 1.0 / (np.exp(exponent) - 1)
        
        # Excited states occupation sum
        terms = (excited_energies - mu) / (k_B * T)
        # Avoid overflow by clipping exponents (terms are positive since excited_energies >= delta > 0 and mu < 0)
        terms = np.clip(terms, None, 500)  # exp(500) is manageable
        n_excited = np.sum(1.0 / (np.exp(terms) - 1))
        
        return n0 + n_excited - N

    # Find mu using Brent's method within a safe bracket
    try:
        # Initial bracket, adjust if necessary based on T
        bracket_low = -1e3 * k_B * T
        bracket_high = -1e-10  # Just below zero
        sol = root_scalar(equation, bracket=[bracket_low, bracket_high], method='brentq')
        mu = sol.root
    except ValueError:
        # Adjust bracket if initial guess fails
        bracket_low = -1e4 * k_B * T
        sol = root_scalar(equation, bracket=[bracket_low, bracket_high], method='brentq')
        mu = sol.root
    except:
        print(f"Root finding failed at T={T}. Skipping.")
        mu = np.nan
        n0 = np.nan
        U = np.nan
        mu_list.append(mu)
        n0_list.append(n0)
        log_n0_list.append(np.log(n0) if n0 > 0 else np.nan)  # Fixed the syntax
        U_list.append(U)

        prev_n0 = n0
        prev_U = U
        prev_T = T
        continue

    # Calculate ground state occupation n0
    exponent = -mu / (k_B * T)
    if exponent < 1e-10:
        n0 = -k_B * T / mu
    else:
        n0 = 1.0 / (np.exp(exponent) - 1)
    
    # Calculate total internal energy U
    terms = (excited_energies - mu) / (k_B * T)
    terms = np.clip(terms, None, 500)
    u_excited = np.sum(excited_energies / (np.exp(terms) - 1))
    U = u_excited  # Ground state energy is 0
    
    # Store results
    mu_list.append(mu)
    n0_list.append(n0)
    log_n0_list.append(np.log(n0) if n0 > 0 else np.nan)
    U_list.append(U)
    
    # Compute derivatives using previous values
    if prev_n0 is not None and prev_T is not None:
        dn0_dT = (n0 - prev_n0) / (T - prev_T)
        dn0_dT_list.append(-dn0_dT)
        
        # Specific heat from energy difference
        if prev_U is not None:
            Cv = (U - prev_U) / (T - prev_T)
            Cv_list.append(Cv)
        else:
            Cv_list.append(np.nan)
    else:
        dn0_dT_list.append(np.nan)
        Cv_list.append(np.nan)
    
    # Update previous values
    prev_n0 = n0
    prev_U = U
    prev_T = T

# Trim the temperature list for derivatives (first point has no derivative)
temps_for_derivs = temps[1:]
dn0_dT_list = dn0_dT_list[:len(temps_for_derivs)]
Cv_list = Cv_list[:len(temps_for_derivs)]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(temps, mu_list)
plt.xlabel('Temperature (T)')
plt.ylabel('Chemical Potential (μ)')
plt.title('Negative Chemical Potential vs Temperature')
plt.grid(True)
plt.savefig('Plots/P2_i1', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(temps, n0_list)
plt.xlabel('Temperature (T)')
plt.ylabel('⟨n₀⟩')
plt.title('Ground State Occupation vs Temperature')
plt.yscale('log')
plt.grid(True)
plt.savefig('Plots/P2_i2', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(temps_for_derivs, dn0_dT_list)
plt.xlabel('Temperature (T)')
plt.ylabel('-∂⟨n₀⟩/∂T')
plt.title('Negative Gradient of Ground State Occupation vs Temperature')
plt.grid(True)
plt.savefig('Plots/P2_i3', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(temps_for_derivs, Cv_list)
plt.xlabel('Temperature (T)')
plt.ylabel('Cv')
plt.title('Specific Heat vs Temperature')
plt.grid(True)
plt.savefig('Plots/P2_i4', bbox_inches='tight')
plt.show()