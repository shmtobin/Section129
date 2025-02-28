# This script demonstrates the derivation and evaluation of the energy absorption per cycle
# of a driven, damped harmonic oscillator. The oscillator obeys the equation:

#     d²x/dt² + γ dx/dt + ω₀² x = F e^(i ω_f t)

# In the frequency domain (via Fourier transform), the response is given by:

#     -ω² x̃ + iγ ω x̃ + ω₀² x̃ = F δ(ω - ω_f)

# Assuming a steady-state solution of the form x(t) = A e^(i ω_f t), we substitute into the
# equation of motion to find the complex amplitude:

#     A = F / (ω₀² - ω_f² + i γ ω_f)

# The amplitude’s magnitude is

#     |A|² = F² / [ (ω₀² - ω_f²)² + (γ ω_f)² ]

# For a damped oscillator, the average power dissipated (which equals the energy absorption per
# unit time) is provided by the damping term. In particular, the average power is given by

#     ⟨P⟩ = γ ⟨(dx/dt)²⟩

# For a sinusoidal steady state, we have

#     ⟨(dx/dt)²⟩ = (1/2) ω_f² |A|²

# so that

#     ⟨P⟩ = (1/2) γ ω_f² |A|².

# Since the period T of the driving force is T = 2π/ω_f, the energy absorbed per cycle is

#     E = ⟨P⟩ T = (1/2) γ ω_f² |A|² * (2π/ω_f)
#       = π γ ω_f |A|².

# Substituting for |A|² we obtain

#     E = (π F² γ ω_f) / [ (ω₀² - ω_f²)² + (γ ω_f)² ].

# This is a Lorentzian function in ω_f. (Note: The result provided in the task statement
# shows a single factor F in the numerator; this discrepancy may be due to different conventions
# for the amplitude F. Here, F is taken as the driving force amplitude.)

import numpy as np
import matplotlib.pyplot as plt

def energy_absorption(omega_f, F, gamma, omega_0):
    """
    Computes the energy absorbed per cycle by a driven, damped harmonic oscillator,
    given by the Lorentzian function:
    
      E = (π * F² * γ * ω_f) / [ (ω₀² - ω_f²)² + (γ * ω_f)² ]
    
    Parameters:
      omega_f: Driving frequency (can be scalar or numpy array)
      F: Amplitude of the driving force
      gamma: Damping coefficient
      omega_0: Natural (resonant) frequency of the oscillator
      
    Returns:
      E: Energy absorbed per cycle
    """
    numerator = np.pi * F**2 * gamma * omega_f
    denominator = (omega_0**2 - omega_f**2)**2 + (gamma * omega_f)**2
    return numerator / denominator

# Example parameters
F = 1.0        # Amplitude of the driving force
gamma = 0.2    # Damping coefficient
omega_0 = 1.0  # Natural frequency of the oscillator

# Create an array of driving frequencies to explore the resonance behavior
omega_f_values = np.linspace(0.1, 2.0, 500)  # Avoid starting at 0 to prevent division by zero

# Calculate the energy absorption per cycle for each driving frequency
E_values = energy_absorption(omega_f_values, F, gamma, omega_0)

# Plot the Lorentzian energy absorption curve
plt.figure(figsize=(8, 5))
plt.plot(omega_f_values, E_values, label='Energy Absorption per Cycle', color='r')
plt.xlabel('Driving Frequency ω_f')
plt.ylabel('Energy Absorbed per Cycle E')
plt.title('Lorentzian Response of a Driven, Damped Harmonic Oscillator')
plt.legend()
plt.grid(True)
plt.show()
