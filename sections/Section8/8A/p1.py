# 8A P1

import numpy as np
import itertools
import matplotlib.pyplot as plt

# ---------------------------- 
# Part a: 2D Lattice generation and Ising Hamiltonian
# ----------------------------

def initialize_lattice(L):
    """Initialize a random LxL lattice with spins {-1, 1}."""
    return np.random.choice([-1, 1], size=(L, L))

def compute_ising_energy(lattice, J=1, B=0):
    """Compute the Ising Hamiltonian for a given lattice configuration."""
    L = lattice.shape[0]
    energy = 0
    for i in range(L):
        for j in range(L):
            S = lattice[i, j]
            # Nearest neighbors (periodic boundary conditions)
            neighbors = lattice[(i+1) % L, j] + lattice[i, (j+1) % L] + \
                        lattice[(i-1) % L, j] + lattice[i, (j-1) % L]
            energy += -J * S * neighbors
    return energy / 2 - B * np.sum(lattice)  # Divide by 2 to avoid double counting

# Example usage
L = 4
lattice = initialize_lattice(L)
energy = compute_ising_energy(lattice)
print("Random Lattice:")
print(lattice)
print("Energy:", energy)

# ---------------------------- 
# Part b: Statistical Description of the Ising Model
# ----------------------------

def compute_partition_function(L, J=1, B=0, T=1):
    """Compute the partition function Z for an LxL lattice by summing over all configurations."""
    beta = 1 / T
    states = list(itertools.product([-1, 1], repeat=L*L))  # Generate all spin configurations
    Z = 0
    energies = []
    
    for state in states:
        lattice = np.array(state).reshape(L, L)
        energy = compute_ising_energy(lattice, J, B)
        energies.append((lattice, energy))
        Z += np.exp(-beta * energy)
    
    return Z, energies

def compute_probabilities(L, J=1, B=0, T=1):
    """Compute the probability mass function P(S) for each spin configuration."""
    Z, energies = compute_partition_function(L, J, B, T)
    beta = 1 / T
    probabilities = [(lattice, np.exp(-beta * energy) / Z) for lattice, energy in energies]
    return probabilities

def visualize_spin_configurations(probabilities):
    """Visualize sampled spin configurations with their probabilities."""
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, (lattice, prob) in enumerate(probabilities[:16]):
        ax = axes[i]
        ax.imshow(lattice, cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f'P={prob:.4f}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('plots/visualize_spin.png', dpi=300)
    plt.show()

# Compute probabilities for L=4, J=1, B=0, T=1
probabilities = compute_probabilities(L=4, J=1, B=0, T=1)
visualize_spin_configurations(probabilities)

# ----------------------------
# Part c: Gibbs Sampler on Ising Model
# ----------------------------
import numpy as np

def compute_energy(lattice, J=1, B=0):
    L = lattice.shape[0]
    energy = 0
    for x in range(L):
        for y in range(L):
            S = lattice[x, y]
            neighbors = lattice[(x+1) % L, y] + lattice[x, (y+1) % L] + lattice[(x-1) % L, y] + lattice[x, (y-1) % L]
            energy += -J * S * neighbors - B * S
    return energy / 2  # Each pair counted twice

def gibbs_sampler(lattice, beta, num_iter=10000):
    L = lattice.shape[0]
    for _ in range(num_iter):
        x, y = np.random.randint(0, L, 2)
        S_new = -lattice[x, y]
        delta_E = 2 * S_new * (lattice[(x+1) % L, y] + lattice[x, (y+1) % L] + lattice[(x-1) % L, y] + lattice[x, (y-1) % L])
        if np.random.rand() < np.exp(-beta * delta_E):
            lattice[x, y] = S_new
    return lattice

# ----------------------------
# Part d: Gibbs Iteration
# ----------------------------
import matplotlib.pyplot as plt

def visualize_gibbs_convergence(L=4, beta=1.0, burn_in=1000, num_samples=5000):
    lattice = np.random.choice([-1, 1], size=(L, L))
    samples = []
    for i in range(burn_in + num_samples):
        lattice = gibbs_sampler(lattice, beta, num_iter=1)
        if i >= burn_in:
            samples.append(np.copy(lattice))
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, idx in enumerate(np.linspace(0, num_samples-1, 5, dtype=int)):
        axes[i].imshow(samples[idx], cmap='gray')
        axes[i].set_title(f'Sample {idx+burn_in}')
        axes[i].axis('off')
    plt.savefig('plots/gibbs_convergence.png', dpi=300)
    plt.show()

visualize_gibbs_convergence()

# ----------------------------
# Part e: Magnetization, Landau Theory, and Phase Transition
# ----------------------------

def compute_magnetization(lattice):
    """Compute the magnetization of a given spin configuration."""
    return np.sum(lattice) / lattice.size

def compute_phase_transition(L, T_values, J=1):
    """Compute the magnetization as a function of temperature."""
    magnetizations = []
    
    for T in T_values:
        beta = 1 / T
        lattice = initialize_lattice(L)
        lattice = gibbs_sampler(lattice, beta, num_iter=5000)  # Burn-in period
        M = compute_magnetization(lattice)
        magnetizations.append(M)
    
    return magnetizations

# Parameters
L = 20  # Lattice size
T_values = np.linspace(1.0, 4.0, 40)  # Temperature range
magnetizations = compute_phase_transition(L, T_values)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(T_values, magnetizations, marker='o', linestyle='-', label='Magnetization')
J = 1  # Default value for the interaction strength

# Calculate critical temperature Tc
Tc = 2 * J / np.log(1 + np.sqrt(2))
plt.axvline(Tc, color='r', linestyle='--', label='Critical Temperature Tc')
plt.xlabel("Temperature")
plt.ylabel("Magnetization")
plt.legend()
plt.title("Phase Transition in 2D Ising Model")
plt.savefig('plots/phase_transition.png', dpi=300)
plt.show()

# ----------------------------
# Part f: Magnetization at Different Temperatures
# ----------------------------

def visualize_magnetization_vs_temperature(lattice_sizes, T_values, J=1):
    """Visualize the phase transition for different lattice sizes."""
    plt.figure(figsize=(8, 5))
    
    for L in lattice_sizes:
        magnetizations = compute_phase_transition(L, T_values, J)
        plt.plot(T_values, magnetizations, marker='o', linestyle='-', label=f'L={L}')
    
    plt.axvline(Tc, color='r', linestyle='--', label='Critical Temperature Tc')
    plt.xlabel("Temperature")
    plt.ylabel("Magnetization")
    plt.legend()
    plt.title("Magnetization vs Temperature for Different Lattice Sizes")
    plt.savefig('plots/magnetization_vs_temperature.png', dpi=300)
    plt.show()


# Parameters
lattice_sizes = [10, 17, 25, 32, 40]
visualize_magnetization_vs_temperature(lattice_sizes, T_values)

# ----------------------------
# Part g: Classical Magnetic Field dependence
# ----------------------------
def magnetization_vs_field(L, T, B_values, steps=10000, burn_in=5000):
    magnetizations = []
    for B in B_values:
        lattice = initialize_lattice(L)
        magnetization = 0
        for step in range(steps):
            i, j = np.random.randint(0, L, size=2)
            delta_E = 2 * lattice[i, j] * (J * (lattice[(i+1)%L, j] + lattice[(i-1)%L, j] +
                                            lattice[i, (j+1)%L] + lattice[i, (j-1)%L]) + B)
            if delta_E <= 0 or np.random.rand() < np.exp(-beta * delta_E):
                lattice[i, j] *= -1
            if step >= burn_in:
                magnetization += np.sum(lattice)
        magnetizations.append(magnetization / ((steps - burn_in) * L**2))
    plt.plot(B_values, magnetizations, label=f"L={L}")
    plt.xlabel("Magnetic Field (B)")
    plt.ylabel("Magnetization")
    plt.legend()
    plt.savefig('plots/magnetization_vs_field.png', dpi=300)
    plt.show()

# ----------------------------
# Part h: Specific Heat of the 2D Ising Model
# ----------------------------
def specific_heat(L, T_values, steps=10000, burn_in=5000):
    specific_heats = []
    for T in T_values:
        beta = 1 / T
        lattice = initialize_lattice(L)
        energy_list = []
        for step in range(steps):
            i, j = np.random.randint(0, L, size=2)
            delta_E = 2 * lattice[i, j] * J * (lattice[(i+1)%L, j] + lattice[(i-1)%L, j] +
                                               lattice[i, (j+1)%L] + lattice[i, (j-1)%L])
            if delta_E <= 0 or np.random.rand() < np.exp(-beta * delta_E):
                lattice[i, j] *= -1
            if step >= burn_in:
                energy_list.append(ising_energy(lattice, J))
        energy_list = np.array(energy_list)
        C_v = (beta**2 / (L**2)) * (np.mean(energy_list**2) - np.mean(energy_list)**2)
        specific_heats.append(C_v)
    plt.plot(T_values, specific_heats, label=f"L={L}")
    plt.xlabel("Temperature (T)")
    plt.ylabel("Specific Heat (C_v)")
    plt.legend()
    plt.savefig('plots/specific_heat.png', dpi=300)
    plt.show()

# ----------------------------
# Part i: Magnetic Susceptibility of the 2D Ising model
# ----------------------------

def compute_magnetic_susceptibility(spin_lattice, beta, L):
    """
    Computes the magnetic susceptibility χ of the 2D Ising model.
    χ = (β / N) * (⟨M^2⟩ - ⟨M⟩^2)
    """
    N = L * L  # Total number of spins
    magnetization = np.sum(spin_lattice)
    M_squared = magnetization ** 2
    avg_M = np.mean(magnetization)
    avg_M_squared = np.mean(M_squared)
    
    susceptibility = (beta / N) * (avg_M_squared - avg_M ** 2)
    return susceptibility

# Example usage:
L = 10  # Lattice size
beta = 1.0  # Inverse temperature
spin_lattice = np.random.choice([-1, 1], size=(L, L))  # Random spin configuration

chi = compute_magnetic_susceptibility(spin_lattice, beta, L)
print(f"Magnetic Susceptibility: {chi}")