# Question 1 Markov chain in site basis

"""
Constructing the Markov chain for a 3-spin Heisenberg XXX Hamiltonian on a ring.

The Hamiltonian is given by:
    H = (3J/4) - J * [ (1/2)(S⁺₁S⁻₂ + S⁻₁S⁺₂) + Sᶻ₁Sᶻ₂ +
                        (1/2)(S⁺₂S⁻₃ + S⁻₂S⁺₃) + Sᶻ₂Sᶻ₃ +
                        (1/2)(S⁺₃S⁻₁ + S⁻₃S⁺₁) + Sᶻ₃Sᶻ₁ ]
with periodic boundary conditions (site 4 is identified with site 1).

The off-diagonal hopping terms (with coefficient -J/2) flip a pair of neighboring spins
when they are anti-aligned:
    |↑↓⟩  <---> |↓↑⟩.

Using the basis (in order):
    0: |↑↑↑⟩
    1: |↑↑↓⟩
    2: |↑↓↑⟩
    3: |↑↓↓⟩
    4: |↓↑↑⟩
    5: |↓↑↓⟩
    6: |↓↓↑⟩
    7: |↓↓↓⟩
we identify the following allowed transitions:

Bond (1,2):
    - From state 2 (|↑↓↑⟩) to state 4 (|↓↑↑⟩) and vice versa.
    - From state 3 (|↑↓↓⟩) to state 5 (|↓↑↓⟩) and vice versa.

Bond (2,3):
    - From state 1 (|↑↑↓⟩) to state 2 (|↑↓↑⟩) and vice versa.
    - From state 5 (|↓↑↓⟩) to state 6 (|↓↓↑⟩) and vice versa.

Bond (3,1) [periodic]:
    - From state 1 (|↑↑↓⟩) to state 4 (|↓↑↑⟩) and vice versa.
    - From state 3 (|↑↓↓⟩) to state 6 (|↓↓↑⟩) and vice versa.

For a Markov chain simulation, we use the absolute value of these matrix elements 
to define transition probabilities. We assume that when a state has allowed moves, 
each allowed move is chosen with equal probability.

Thus:
    - States with no moves (state 0 and state 7) have P(i -> i) = 1.
    - For example, state 1 (|↑↑↓⟩) connects to state 2 and state 4, so:
          P(1 -> 2) = 1/2,  P(1 -> 4) = 1/2.
"""

import numpy as np

# Number of states in the 3-spin system
n_states = 8

# Initialize the transition matrix with zeros
P = np.zeros((n_states, n_states))

# For clarity, define the state indices corresponding to our basis:
# 0: |↑↑↑⟩, 1: |↑↑↓⟩, 2: |↑↓↑⟩, 3: |↑↓↓⟩,
# 4: |↓↑↑⟩, 5: |↓↑↓⟩, 6: |↓↓↑⟩, 7: |↓↓↓⟩

# Allowed transitions based on the hopping terms:

# Bond (1,2):
#   Transition between state 2 and state 4:
P[2,4] = 1/2  # from state 2 to state 4
P[4,2] = 1/2  # from state 4 to state 2
#   Transition between state 3 and state 5:
P[3,5] = 1/2  # from state 3 to state 5
P[5,3] = 1/2  # from state 5 to state 3

# Bond (2,3):
#   Transition between state 1 and state 2:
P[1,2] = 1/2  # from state 1 to state 2
P[2,1] = 1/2  # from state 2 to state 1
#   Transition between state 5 and state 6:
P[5,6] = 1/2  # from state 5 to state 6
P[6,5] = 1/2  # from state 6 to state 5

# Bond (3,1) [periodic]:
#   Transition between state 1 and state 4:
P[1,4] = 1/2  # from state 1 to state 4
P[4,1] = 1/2  # from state 4 to state 1
#   Transition between state 3 and state 6:
P[3,6] = 1/2  # from state 3 to state 6
P[6,3] = 1/2  # from state 6 to state 3

# For states that do not have any allowed moves under the off-diagonal (hopping) terms,
# we let them be absorbing (i.e. self-transition probability is 1).
# These are the ferromagnetic states: state 0 (|↑↑↑⟩) and state 7 (|↓↓↓⟩).
P[0,0] = 1.0
P[7,7] = 1.0

# Note: States with allowed moves have no self-transition probability (P(i->i)=0)
# because we assume that a move is always attempted when available.
# (In a more general Metropolis or heat-bath simulation, one might also include rejection moves.)

# Print the Markov chain transition probability matrix
np.set_printoptions(precision=2, suppress=True)
print("Markov chain transition probability matrix P:")
print(P)

# Question 2 Markov chain in site basis

"""
This script demonstrates that the stationary distribution π for a 3-spin
Heisenberg XXX model (in the S^z basis) can be obtained both by solving the
eigenvalue problem πP = π (with ∑_i π_i = 1) for a properly constructed
Markov chain and by using the Boltzmann weights.

We represent each state as a tuple of three spins (each ±1, corresponding to S^z = ±1/2).
The energy (from the diagonal part of the Hamiltonian) is given by:
    E = (3J/4) - (J/4)*(s0*s1 + s1*s2 + s2*s0)
For J = 1, the fully aligned states have E = 0 and the remaining six states have E = 1.

We then construct a Markov chain using single-spin flip Metropolis moves:
  - For a move i -> j (which flips one spin), the proposal probability is 1/3.
  - The acceptance probability is a = min{1, exp[-β*(E(j)-E(i))]}.
The resulting transition matrix P satisfies detailed balance with respect to the Boltzmann
distribution:
    π_i ∝ exp(-β E_i).

Finally, we verify that:
  (i)  π P ≈ π, and
  (ii) ∑_i π_i = 1.
"""

import numpy as np
from itertools import product
from numpy.linalg import eig

# Parameters
J = 1.0      # Coupling constant
T = 1.0      # Temperature (with k_B = 1)
beta = 1.0 / T

# Define all states of the 3-spin chain as tuples of ±1 (representing spin up/down)
states = list(product([1, -1], repeat=3))  # There are 8 states
n_states = len(states)

def energy(state, J=J):
    """
    Compute the energy of a state given by a tuple of three spins (each ±1).
    Using:
      E = (3J/4) - (J/4)*(s0*s1 + s1*s2 + s2*s0)
    """
    s0, s1, s2 = state
    return (3*J/4) - (J/4)*(s0*s1 + s1*s2 + s2*s0)

# Compute energies and Boltzmann weights for each state
energies = np.array([energy(state) for state in states])
weights = np.exp(-beta * energies)
Z = np.sum(weights)
pi_boltzmann = weights / Z

print("States and their energies:")
for i, state in enumerate(states):
    print(f"State {i:1d} {state}: E = {energies[i]:.2f}")

print("\nBoltzmann stationary distribution (π ∝ exp(-βE)):")
for i, p in enumerate(pi_boltzmann):
    print(f"State {i:1d}: π = {p:.4f}")
print(f"Partition function Z = {Z:.4f}\n")

# Construct the Markov chain transition matrix P using single-spin flip Metropolis moves.
# For each state, list neighbors obtained by flipping one spin.
P = np.zeros((n_states, n_states))
for i, state in enumerate(states):
    neighbors = []
    # Identify all neighbors: flip each spin
    for j in range(3):
        new_state = list(state)
        new_state[j] *= -1  # flip the j-th spin
        new_state = tuple(new_state)
        # Find the index of the new state
        k = states.index(new_state)
        neighbors.append(k)
    # For each neighbor, compute the acceptance probability
    total_move_prob = 0.0
    for k in neighbors:
        dE = energies[k] - energies[i]
        a = min(1.0, np.exp(-beta * dE))  # Metropolis acceptance probability
        P[i, k] = (1.0/3.0) * a  # proposal probability is 1/3
        total_move_prob += (1.0/3.0) * a
    # Self-transition probability: if move is rejected or no move is made, remain in state i.
    P[i, i] = 1.0 - total_move_prob

# Verify that each row of P sums to 1
row_sums = P.sum(axis=1)
print("Row sums of P (should all be 1):")
print(np.round(row_sums, 6))
print()

print("Markov chain transition matrix P:")
print(np.round(P, 4))
print()

# Now, solve for the stationary distribution π by finding the left eigenvector of P with eigenvalue 1.
# That is, find π such that πP = π.
eigvals, eigvecs = eig(P.T)
# Find the eigenvector corresponding to eigenvalue 1 (within numerical tolerance)
idx = np.argmin(np.abs(eigvals - 1))
pi_eig = np.real(eigvecs[:, idx])
# Normalize the eigenvector so that sum(pi) = 1
pi_eig = pi_eig / np.sum(pi_eig)

print("Stationary distribution from solving πP = π:")
for i, p in enumerate(pi_eig):
    print(f"State {i:1d}: π = {p:.4f}")

print("\nDifference between Boltzmann π and eigenvector π:")
print(np.round(pi_boltzmann - pi_eig, 6))
print()

# Check that πP ≈ π:
piP = pi_eig @ P
print("π P (using eigenvector π):")
print(np.round(piP, 6))
print()
print("Difference πP - π:")
print(np.round(piP - pi_eig, 6))

# Question 3  Markov chain in site basis

"""
This script constructs the Markov chain for a 3-spin Heisenberg XXX model 
(in the S^z basis) and uses the power iteration method

    π_{k+1} = π_k P

to find its stationary distribution starting from three different initial guesses:

  1) All probability in state |↑↑↑⟩.
  2) 50% probability in state |↑↑↑⟩ and 50% in state |↓↑↓⟩.
  3) A uniformly distributed initial configuration.

We then verify that the resulting stationary distribution satisfies:
    πP = π   and   Σ_i π_i = 1.
    
The 3-spin states are represented as tuples of three numbers (each ±1, corresponding to S^z = ±½),
with the ordering:
  0: |↑↑↑⟩   (i.e. (1,1,1))
  1: |↑↑↓⟩   (i.e. (1,1,-1))
  2: |↑↓↑⟩   (i.e. (1,-1,1))
  3: |↑↓↓⟩   (i.e. (1,-1,-1))
  4: |↓↑↑⟩   (i.e. (-1,1,1))
  5: |↓↑↓⟩   (i.e. (-1,1,-1))
  6: |↓↓↑⟩   (i.e. (-1,-1,1))
  7: |↓↓↓⟩   (i.e. (-1,-1,-1))

The energy (diagonal part) for a state is given by:
    E = (3J/4) - (J/4)*(s0*s1 + s1*s2 + s2*s0)
For J=1, the fully aligned states (0 and 7) have E = 0 and the remaining states have E = 1.
The Metropolis algorithm is used to construct the transition matrix P.
"""

import numpy as np
from itertools import product

# Parameters
J = 1.0      # Coupling constant
T = 1.0      # Temperature (units where k_B = 1)
beta = 1.0/T

# Generate all states for a 3-spin chain (each spin: +1 (up) or -1 (down))
states = list(product([1, -1], repeat=3))
n_states = len(states)

def energy(state, J=J):
    """
    Compute the energy of a state (a tuple of three spins)
    using the diagonal part of the Hamiltonian:
      E = (3J/4) - (J/4) * (s0*s1 + s1*s2 + s2*s0)
    """
    s0, s1, s2 = state
    return (3 * J / 4) - (J / 4) * (s0 * s1 + s1 * s2 + s2 * s0)

# Compute energies for all states
energies = np.array([energy(s) for s in states])

# Construct the transition matrix P using single-spin flip Metropolis moves.
# For each state, consider flipping each of the 3 spins.
P = np.zeros((n_states, n_states))
for i, state in enumerate(states):
    neighbors = []
    for j in range(3):
        new_state = list(state)
        new_state[j] *= -1  # flip the j-th spin
        new_state = tuple(new_state)
        k = states.index(new_state)
        neighbors.append(k)
    total_move_prob = 0.0
    # For each neighbor, compute the Metropolis acceptance probability.
    for k in neighbors:
        dE = energies[k] - energies[i]
        a = min(1.0, np.exp(-beta * dE))
        P[i, k] = (1.0 / 3.0) * a  # proposal probability is 1/3 for each spin flip
        total_move_prob += (1.0 / 3.0) * a
    # Set the self-transition probability so the row sums to 1.
    P[i, i] = 1.0 - total_move_prob

# Verify that each row sums to 1.
print("Row sums of P (should be 1):")
print(np.round(P.sum(axis=1), 6))
print()

# Define a function to perform power iteration.
def power_iteration(pi0, P, tol=1e-10, max_iter=10000):
    """
    Use power iteration to compute the stationary distribution.
    Starting from initial guess pi0, iterate: pi = pi @ P
    until convergence.
    """
    pi = pi0.copy()
    for k in range(max_iter):
        new_pi = pi @ P
        if np.linalg.norm(new_pi - pi, ord=1) < tol:
            return new_pi, k
        pi = new_pi
    return pi, max_iter

# --- Initial Guesses ---

# 1) All probability in state |↑↑↑⟩ (state 0).
pi0_1 = np.zeros(n_states)
pi0_1[0] = 1.0

# 2) 50% probability in |↑↑↑⟩ (state 0) and 50% in |↓↑↓⟩.
# Note: |↓↑↓⟩ corresponds to state 5 (i.e. (-1, 1, -1)).
pi0_2 = np.zeros(n_states)
pi0_2[0] = 0.5
pi0_2[5] = 0.5

# 3) Uniform distribution over all states.
pi0_3 = np.ones(n_states) / n_states

# Compute stationary distributions using power iteration for each initial guess.
pi1, iter1 = power_iteration(pi0_1, P)
pi2, iter2 = power_iteration(pi0_2, P)
pi3, iter3 = power_iteration(pi0_3, P)

# Print the results.
print("Stationary distribution using power iteration:")
print("Initial guess 1 (|↑↑↑⟩ = 1):")
print(np.round(pi1, 6))
print("Iterations:", iter1)
print()

print("Initial guess 2 (|↑↑↑⟩ = 0.5, |↓↑↓⟩ = 0.5):")
print(np.round(pi2, 6))
print("Iterations:", iter2)
print()

print("Initial guess 3 (Uniform distribution):")
print(np.round(pi3, 6))
print("Iterations:", iter3)
print()

# Verify that each stationary distribution satisfies πP ≈ π and that ∑π_i = 1.
for idx, (pi_final, guess) in enumerate(zip([pi1, pi2, pi3], [1,2,3]), start=1):
    print(f"Verification for initial guess {guess}:")
    print("  Sum of π =", np.sum(pi_final))
    piP = pi_final @ P
    diff = np.linalg.norm(piP - pi_final, ord=1)
    print("  ||πP - π||₁ =", diff)
    print()

# Question 4 Markov chain in magnon basis


"""
Markov Chain in Magnon Basis for N=3

In the magnon formulation the ground state (ferromagnetic vacuum) is defined as
    |0⟩ = |↑↑↑⟩,
with energy H|0⟩ = 0.

A one-magnon state is defined as:
    |p⟩ = ∑ₙ e^(ipn) S⁻ₙ|0⟩,
with periodic boundary conditions enforcing
    e^(ipN)=1  ⇒  p=2πk/N,  k=0,1,2,...,N-1.

For N=3 the allowed momenta and corresponding energies are:
    p₀ = 0,           E₀ = 2J sin²(0/2)= 0,
    p₁ = 2π/3,        E₁ = 2J sin²(π/3)= 3J/2,
    p₂ = 4π/3,        E₂ = 3J/2   (degenerate with k=1).

Assuming Boltzmann-type transitions between magnons, we adopt a Metropolis-type rule.
That is, for a proposal to change from |k⟩ to |k'⟩ (with a uniform proposal probability of 1/2),
we set
    a(k→k') = min{1, exp[-β (E_{k'} - E_k)]}.

Thus, for i≠j, we define:
    P_{ij} = (1/2) · a(i→j),
and then set the diagonal element such that the row sums to one:
    P_{ii} = 1 - ∑_{j≠i} P_{ij}.

We also note that in the magnon basis the equilibrium distribution is:
    π_k ∝ exp(-β E_k),
which, after normalization, satisfies πP = π and ∑ₖ π_k = 1.

Finally, note the differences:
 - In the site basis the Markov chain is defined on 8 states (for 3 spins) with local moves (spin flips)
   and a sparse connectivity reflecting the underlying lattice.
 - In the magnon basis the chain is defined on 3 collective (momentum) states and the transition probabilities
   depend only on the energy differences as given by the dispersion relation. This leads to a much smaller,
   fully-connected matrix.
"""

import numpy as np

# Parameters
J = 1.0       # Coupling constant (energy scale)
T = 1.0       # Temperature (with k_B = 1)
beta = 1.0/T

# Number of magnon states for N=3
N_m = 3

# Allowed magnon states: k = 0, 1, 2.
# Their energies are given by:
# E_k = 2J sin^2( (pi * k)/3 )
# (since p_k = 2pi*k/3 and sin(p_k/2) = sin(pi*k/3))
energies = np.array([2*J*np.sin(np.pi*k/3)**2 for k in range(N_m)])
# For J=1, this gives: E0 = 0, E1 = 2*sin^2(pi/3)=2*(3/4)=1.5, E2 = 1.5.
print("Magnon energies for N=3:")
for k, E in enumerate(energies):
    print(f"  k = {k}: E = {E:.4f}")

# Compute Boltzmann weights and equilibrium distribution in magnon basis.
weights = np.exp(-beta * energies)
Z = np.sum(weights)
pi_eq = weights / Z

print("\nEquilibrium (Boltzmann) distribution in magnon basis:")
for k, p in enumerate(pi_eq):
    print(f"  State |{k}⟩: π = {p:.4f}")
print(f"  Partition function Z = {Z:.4f}\n")

# Construct the transition matrix P in the magnon basis using a Metropolis rule.
# Proposal: from any state, propose a transition to any of the other two with equal probability 1/2.
P = np.zeros((N_m, N_m))
for i in range(N_m):
    total_move_prob = 0.0
    for j in range(N_m):
        if i == j:
            continue
        dE = energies[j] - energies[i]
        # Metropolis acceptance probability:
        a = min(1.0, np.exp(-beta * dE))
        P[i, j] = (1/2) * a   # proposal probability is 1/2
        total_move_prob += P[i, j]
    # Self-transition probability so that the row sums to 1.
    P[i, i] = 1.0 - total_move_prob

print("Magnon basis transition matrix P (Metropolis rule):")
print(np.round(P, 4))
print()

# (Optional) Verify that each row sums to 1.
print("Row sums of P (should be 1):", np.round(P.sum(axis=1), 6))
print()

# Let’s briefly compare to what we expect:
# For state 0 (E0=0):
#  - Transition to state 1: dE = 1.5, acceptance = exp(-1.5) ~ 0.22313, so P[0,1] ~ 0.111565.
#  - Similarly, P[0,2] ~ 0.111565, and P[0,0] ~ 1 - 0.22313 = 0.77687.
# For states 1 and 2 (E1=E2=1.5):
#  - Transition from state 1 to state 0: dE = -1.5, acceptance = 1, so P[1,0] = 0.5.
#  - Transition from state 1 to state 2: dE = 0, acceptance = 1, so P[1,2] = 0.5, and P[1,1] = 0.
#  - Same for state 2.
#
# In contrast, in the site basis (with 8 states) the connectivity is sparse and only local moves are allowed.
# Here, in the magnon basis, every state (being a collective excitation) is connected to every other,
# and the transition probabilities depend solely on the energy differences given by the dispersion relation.

# One can check that if one computes the stationary distribution from P (e.g., by power iteration or
# solving πP = π), one obtains the Boltzmann distribution π_eq.
# For example:
pi_stat = np.linalg.eig(P.T)[1][:, np.argmax(np.isclose(np.linalg.eig(P.T)[0], 1))]
pi_stat = np.real(pi_stat / np.sum(pi_stat))
print("Stationary distribution from solving πP = π (magneton basis):")
for k, p in enumerate(pi_stat):
    print(f"  State |{k}⟩: π = {p:.4f}")

# In summary, the main differences between the two bases are:
# 1. The site basis Markov chain is defined on 8 states with transitions given by local (spin flip)
#    moves; its matrix is sparse and reflects the geometry of the ring.
# 2. The magnon basis chain is defined on 3 collective (momentum) states with transitions
#    given by Boltzmann-type factors based on the dispersion relation. The connectivity is global,
#    and the transition probabilities depend solely on energy differences.

# Question 5 Markov chain in magnon basis


"""
Markov Chain in Magnon Basis: Stationary Distribution vs Temperature

We consider the 3-magnon basis (N=3) for a ferromagnetic ring.
Allowed momentum states:
   |k⟩, with k = 0, 1, 2,
and quantized momenta:
   p_k = 2πk/3.
The dispersion relation gives the magnon energies:
   E_k = 2J sin²( (πk)/3 ).
For J=1:
   E₀ = 0,  E₁ = E₂ = 1.5.
   
We assume Boltzmann-type transitions between magnon states.
For i ≠ j, using a Metropolis rule with a uniform proposal probability of 1/2:
   P_{ij} = (1/2) * min{1, exp[-β(E_j - E_i)]},
and then we set
   P_{ii} = 1 - ∑_{j≠i} P_{ij},
so that each row of P sums to 1.

The stationary distribution is given by π satisfying
   πP = π  and  ∑_k π_k = 1.
At equilibrium π_k = exp(-βE_k)/Z.

This script calculates π for various temperatures and shows that:
  - At low temperature, the vacuum (|0⟩) dominates.
  - At high temperature, π approaches a uniform distribution.
It also highlights the differences compared to the site basis, where the Markov chain is defined on 8 local states.
"""

import numpy as np

def magnon_transition_matrix(J, T):
    """
    Constructs the 3x3 transition matrix P in the magnon basis for N=3.
    
    Args:
        J: Coupling constant.
        T: Temperature (with k_B=1).
        
    Returns:
        P: 3x3 transition matrix.
        energies: Array of magnon energies [E0, E1, E2].
    """
    beta = 1.0 / T
    N_m = 3
    # Compute magnon energies: E_k = 2J * sin²(π*k/3)
    energies = np.array([2 * J * (np.sin(np.pi * k / 3)**2) for k in range(N_m)])
    
    # Initialize P
    P = np.zeros((N_m, N_m))
    
    # For each state i, compute off-diagonal elements:
    for i in range(N_m):
        total_move_prob = 0.0
        for j in range(N_m):
            if i == j:
                continue
            dE = energies[j] - energies[i]
            a = min(1.0, np.exp(-beta * dE))  # Metropolis acceptance probability
            P[i, j] = 0.5 * a  # Proposal probability is 1/2 (since there are 2 choices)
            total_move_prob += P[i, j]
        P[i, i] = 1.0 - total_move_prob  # Ensure row sum equals 1
    return P, energies

def stationary_distribution(P):
    """
    Computes the stationary distribution as the left eigenvector of P with eigenvalue 1.
    """
    eigvals, eigvecs = np.linalg.eig(P.T)
    # Find eigenvector corresponding to eigenvalue 1 (within numerical tolerance)
    idx = np.argmin(np.abs(eigvals - 1))
    pi = np.real(eigvecs[:, idx])
    return pi / np.sum(pi)

def boltzmann_distribution(energies, T):
    """
    Computes the Boltzmann distribution for given energies at temperature T.
    """
    beta = 1.0 / T
    weights = np.exp(-beta * energies)
    return weights / np.sum(weights)

# Coupling constant
J = 1.0

# Temperatures to consider
temperatures = [0.1, 0.5, 1.0, 2.0, 10.0]

print("Magnon Basis Stationary Distribution vs Temperature (N=3, J=1):")
for T in temperatures:
    P, energies = magnon_transition_matrix(J, T)
    pi_stat = stationary_distribution(P)
    pi_boltz = boltzmann_distribution(energies, T)
    
    print(f"\nTemperature T = {T:.2f}:")
    print("Magnon energies: ", np.round(energies, 4))
    
    print("Stationary distribution from solving πP = π:")
    for k, p in enumerate(pi_stat):
        print(f"  |{k}⟩: π = {p:.4f}")
        
    print("Boltzmann distribution (expected):")
    for k, p in enumerate(pi_boltz):
        print(f"  |{k}⟩: π = {p:.4f}")
        
    # Verify normalization and stationarity:
    print("Sum of π =", np.sum(pi_stat))
    diff = np.linalg.norm(pi_stat @ P - pi_stat, ord=1)
    print("||πP - π||₁ =", diff)

print("\nDiscussion:")
print("1. At low temperatures (e.g., T = 0.1), the ground state |0⟩ (with E = 0) dominates,")
print("   since exp(-βE) for E = 1.5 is very small. Hence, almost all the probability is in |0⟩.")
print("2. As T increases, the difference between exp(-β*0) and exp(-β*1.5) diminishes,")
print("   so the excited states |1⟩ and |2⟩ become more populated. In the high-T limit,")
print("   all states become equally probable (uniform distribution).")
print("3. In comparison to the site basis:")
print("   - The magnon basis involves only 3 collective states (momentum eigenstates),")
print("     and the connectivity of the Markov chain is global (any magnon can transition to any other).")
print("   - The site basis has 8 states with local spin-flip moves, leading to a sparse transition matrix,")
print("     and additional degeneracies and spatial structure influence the stationary distribution.")

# Question 6 Markov chain in magnon basis

#!/usr/bin/env python3
"""
Markov Chain in Magnon Basis using Power Iteration

We consider a magnon formulation for a ferromagnetic ring with N=8 sites.
The allowed one–magnon states are:
   |k⟩,  with k = 0, 1, ..., 7,
with quantized momenta p = 2πk/N.
The energy dispersion is given by:
   E_k = 2J sin²(π k / N).
For J = 1 (and k_B = 1), these energies are:
   E_0 = 0,
   E_1 = 2 sin²(π/8),
   E_2 = 2 sin²(π/4),
   E_3 = 2 sin²(3π/8),
   E_4 = 2 sin²(π/2) = 2,
   E_5 = E_3,
   E_6 = E_2,
   E_7 = E_1.
We assume Boltzmann‐type transitions between magnons.
For i ≠ j, the transition probability is defined as:
   P_{ij} = (1/(N-1)) * min{1, exp[-β (E_j - E_i)]},
and we choose P_{ii} so that each row sums to 1.

We then apply the power iteration method:
   π⁽ᵏ⁺¹⁾ = π⁽ᵏ⁾ P,
with the following three initial guesses:
  1) All probability in |k=1⟩.
  2) 50% in |k=1⟩ and 50% in |k=4⟩.
  3) Uniform distribution over all states.

At convergence, π should satisfy πP = π and ∑π_k = 1.
Furthermore, it should coincide with the Boltzmann distribution:
   π_k = exp(-β E_k) / Z.
"""

import numpy as np

# Parameters
J = 1.0          # Coupling constant
T = 1.0          # Temperature (with k_B = 1)
beta = 1.0 / T
N = 8            # Number of magnon states (sites)

# Compute magnon energies: E_k = 2J sin^2(π*k/N)
energies = np.array([2 * J * np.sin(np.pi * k / N)**2 for k in range(N)])
print("Magnon energies (for N=8, J=1):")
for k, E in enumerate(energies):
    print(f"  k = {k}: E = {E:.4f}")

# Build the transition matrix P in the magnon basis using a Metropolis-type rule.
# Proposal probability: 1/(N-1) for any j ≠ i.
P = np.zeros((N, N))
proposal_prob = 1.0 / (N - 1)
for i in range(N):
    total = 0.0
    for j in range(N):
        if i == j:
            continue
        dE = energies[j] - energies[i]
        acceptance = min(1.0, np.exp(-beta * dE))
        P[i, j] = proposal_prob * acceptance
        total += P[i, j]
    # Diagonal element chosen to make the row sum to 1.
    P[i, i] = 1.0 - total

# Check that rows sum to 1
print("\nTransition matrix P (magnon basis):")
print(np.round(P, 4))
print("Row sums:", np.round(P.sum(axis=1), 6))

# Define the power iteration function
def power_iteration(pi0, P, tol=1e-12, max_iter=10000):
    pi = pi0.copy()
    for k in range(max_iter):
        new_pi = pi @ P
        if np.linalg.norm(new_pi - pi, ord=1) < tol:
            return new_pi, k+1
        pi = new_pi
    return pi, max_iter

# Expected Boltzmann stationary distribution
boltzmann_weights = np.exp(-beta * energies)
Z = np.sum(boltzmann_weights)
pi_boltz = boltzmann_weights / Z

# Initial guesses:
# 1) All probability in |k=1⟩.
pi0_1 = np.zeros(N)
pi0_1[1] = 1.0

# 2) 50% probability in |k=1⟩ and 50% in |k=4⟩.
pi0_2 = np.zeros(N)
pi0_2[1] = 0.5
pi0_2[4] = 0.5

# 3) Uniform distribution.
pi0_3 = np.ones(N) / N

# Perform power iteration for each initial guess.
pi_stat1, iter1 = power_iteration(pi0_1, P)
pi_stat2, iter2 = power_iteration(pi0_2, P)
pi_stat3, iter3 = power_iteration(pi0_3, P)

# Print results
print("\nStationary distribution from power iteration (magnon basis):")
print("Initial guess 1: All probability in |k=1⟩")
print(np.round(pi_stat1, 6))
print("Iterations:", iter1)

print("\nInitial guess 2: 50% in |k=1⟩ and 50% in |k=4⟩")
print(np.round(pi_stat2, 6))
print("Iterations:", iter2)

print("\nInitial guess 3: Uniform distribution")
print(np.round(pi_stat3, 6))
print("Iterations:", iter3)

print("\nExpected Boltzmann distribution (for T=1):")
for k, p in enumerate(pi_boltz):
    print(f"  |{k}⟩: π = {p:.6f}")

# Verify stationarity: πP should equal π.
print("\nVerification: For guess 1, πP - π (L1 norm) =", np.linalg.norm(pi_stat1 @ P - pi_stat1, ord=1))


# Question 7 Master equation evolution

"""
Master Equation Evolution in the Magnon Basis

We start from the discrete–time Markov chain in the magnon basis. For a ring with N=8,
the one–magnon states |k⟩ (with k=0,1,…,7) have energies (with J=1):
    E_k = 2J sin^2(π*k/N).

We previously constructed a transition matrix P (using a Metropolis rule) with the form:
    P_{ij} = (1/(N-1)) * min{1, exp[-β (E_j - E_i)]}  for i≠j,
with diagonal elements chosen so that each row sums to 1.
    
To obtain a continuous–time master equation,
    dπ/dt = π Q,
we use the relation:
    P^n ≈ exp(Q (n Δt)),
so that
    Q ≈ (1/(n Δt)) * ln(P^n).

In this script:
  - We construct the magnon transition matrix P (with N=8, T=1, J=1).
  - We choose n = 10 and Δt = 1, and compute Q.
  - We then solve the ODE:
         dπ/dt = π Q
    with the initial condition π(0) = [0, 1, 0, 0, 0, 0, 0, 0] (i.e. all probability in |k=1⟩).
  - Finally, we visualize the evolution of the probability π_i(t) for each state.
    
Note: In a continuous–time Markov chain generator Q, the off–diagonal entries are nonnegative and 
each row sums to 0.
"""

import numpy as np
from scipy.linalg import logm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
J = 1.0         # Coupling constant
T = 1.0         # Temperature (with k_B = 1)
beta = 1.0 / T
N = 8           # Number of magnon states
delta_t = 1.0   # Discrete time step for P
n_steps = 10    # We'll use P^n with n = 10

# Construct magnon energies: E_k = 2J * sin^2(π*k/N)
energies = np.array([2 * J * (np.sin(np.pi * k / N)**2) for k in range(N)])
print("Magnon energies:")
for k, E in enumerate(energies):
    print(f"  k = {k}: E = {E:.4f}")

# Build the transition matrix P in the magnon basis using a Metropolis-type rule.
# For i ≠ j: proposal probability = 1/(N-1), acceptance = min{1, exp[-β (E_j - E_i)]}.
P = np.zeros((N, N))
proposal_prob = 1.0 / (N - 1)
for i in range(N):
    total = 0.0
    for j in range(N):
        if i == j:
            continue
        dE = energies[j] - energies[i]
        acceptance = min(1.0, np.exp(-beta * dE))
        P[i, j] = proposal_prob * acceptance
        total += P[i, j]
    # Set diagonal so that row i sums to 1.
    P[i, i] = 1.0 - total

print("\nDiscrete-time transition matrix P (magnon basis):")
print(np.round(P, 4))
print("Row sums:", np.round(P.sum(axis=1), 6))

# Convert P into the continuous-time rate matrix Q.
# We use: Q = (1/(n*delta_t)) * logm(P^n), with n = n_steps.
P_n = np.linalg.matrix_power(P, n_steps)
Q = logm(P_n) / (n_steps * delta_t)
# For real Q, take the real part (numerical round-off may introduce a tiny imaginary part).
Q = np.real(Q)
print("\nContinuous-time rate matrix Q (computed from P^n):")
print(np.round(Q, 4))
# Check that rows of Q sum to (approximately) 0:
print("Row sums of Q (should be ~0):", np.round(Q.sum(axis=1), 6))

# Define the continuous-time master equation: dπ/dt = π Q.
# Here π is a row vector. The ODE function returns dπ/dt.
def master_equation(t, pi):
    return pi @ Q

# Initial condition: all probability in |k=1⟩ (index 1).
pi0 = np.zeros(N)
pi0[1] = 1.0

# Time span for the integration
t_span = (0, 20)  # integrate from t=0 to t=20 (arbitrary units)
t_eval = np.linspace(t_span[0], t_span[1], 400)

# Solve the ODE using a SciPy integrator.
sol = solve_ivp(master_equation, t_span, pi0, t_eval=t_eval, method='RK45')

# Check that the solution remains normalized.
norm = sol.y.sum(axis=0)
print("\nNormalization check (should be ~1):")
print(np.round(norm, 6))

# Plot the probability evolution for each magnon state.
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(sol.t, sol.y[i, :], label=f"|k={i}⟩")
plt.xlabel("Time t")
plt.ylabel("Probability π_i(t)")
plt.title("Evolution of the Magnon Population Probabilities")
plt.legend()
plt.grid(True)
plt.show()
