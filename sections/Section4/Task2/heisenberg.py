#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------
# Question 1: Construct the Heisenberg XXX Hamiltonian with periodic boundary conditions
# ----------------------------
def construct_heisenberg_hamiltonian(N, J):
    """
    Constructs the Hamiltonian matrix for the Heisenberg XXX chain on a ring.
    
    The Hamiltonian is given by:
    
      H = (J/4)*N*I - (J/2) * Σ_{i=0}^{N-1} ( S⁺_i S⁻_{i+1} + S⁻_i S⁺_{i+1} )
          - J * Σ_{i=0}^{N-1} (Sz_i * Sz_{i+1})
    
    with periodic boundary condition (site N is identified with site 0).
    
    In our representation:
      - A basis state is an integer from 0 to 2^N - 1.
      - Bit i = 1 means spin up (S_z = +1/2), and bit i = 0 means spin down (S_z = -1/2).
    
    For each bond between sites i and j=(i+1)%N:
      - Diagonal contribution:
          * Constant shift: J/4
          * Sz interaction: -J * (S_z(i)*S_z(j))
        Thus, if spins are opposite (S_z = +1/2 and -1/2), the total contribution is J/4 + J/4 = J/2.
        If spins are the same, it cancels (J/4 - J/4 = 0).
      - Off-diagonal contribution:
          * If spin i is up and spin j is down, then S⁻_i S⁺_j flips them, adding –J/2.
          * If spin i is down and spin j is up, then S⁺_i S⁻_j flips them, adding –J/2.
    
    Note: The full Hilbert space has dimension 2^N and the algorithm scales as O(N * 2^N).
    """
    dim = 2**N
    H = np.zeros((dim, dim), dtype=float)
    
    # Loop over each basis state (represented by an integer)
    for state in range(dim):
        # For each bond (with periodic boundary conditions)
        for i in range(N):
            j = (i + 1) % N  # neighbor index (wraps around)
            # Get the spin at site i and j:
            # bit = 1  => spin up => S_z = +1/2
            # bit = 0  => spin down => S_z = -1/2
            si = 0.5 if ((state >> i) & 1) else -0.5
            sj = 0.5 if ((state >> j) & 1) else -0.5
            
            # Diagonal contribution: constant shift plus Sz_i * Sz_j term.
            # If spins are opposite, contribution = J/4 + J/4 = J/2; if same, it cancels.
            if si != sj:
                H[state, state] += J / 2
            else:
                H[state, state] += 0  # (J/4 - J/4 = 0)
            
            # Off-diagonal contribution: flip the spins if they are opposite.
            # For state with spin configuration at sites i and j:
            if si == 0.5 and sj == -0.5:
                # Flip: site i goes from up to down, site j goes from down to up.
                new_state = state
                new_state = new_state & ~(1 << i)  # set bit i to 0 (down)
                new_state = new_state | (1 << j)     # set bit j to 1 (up)
                H[new_state, state] += -J / 2
            elif si == -0.5 and sj == 0.5:
                # Flip: site i goes from down to up, site j goes from up to down.
                new_state = state
                new_state = new_state | (1 << i)     # set bit i to 1 (up)
                new_state = new_state & ~(1 << j)    # set bit j to 0 (down)
                H[new_state, state] += -J / 2
    return H

# ----------------------------
# Question 2: Diagonalize the Hamiltonian using the QR algorithm
# ----------------------------
def qr_diagonalize(H, tol=1e-8, max_iter=1000):
    """
    Diagonalizes a matrix H using the iterative QR algorithm.
    
    The algorithm:
      1. Set A = H.
      2. For each iteration, compute the QR decomposition A = Q R.
      3. Update A = R Q.
      4. Stop when the off-diagonal Frobenius norm is below tol.
    
    Returns:
      The approximately diagonal matrix A whose diagonal elements are the eigenvalues.
    """
    A = H.copy()
    for k in range(max_iter):
        Q, R = np.linalg.qr(A)
        A_new = R @ Q
        # Measure off-diagonal norm:
        off_diag_norm = np.linalg.norm(A_new - np.diag(np.diag(A_new)), ord='fro')
        if off_diag_norm < tol:
            print(f"QR algorithm converged after {k+1} iterations")
            return A_new
        A = A_new
    print("QR algorithm did not converge within the maximum iterations")
    return A

# ----------------------------
# Main: Testing and time complexity analysis
# ----------------------------
if __name__ == '__main__':
    J = 1.0  # coupling constant
    # Choose a test value for N (number of spins)
    N_test = 4
    H = construct_heisenberg_hamiltonian(N_test, J)
    print(f"Hamiltonian for N = {N_test} spins (dimension = {2**N_test}):")
    print(H)
    
    # Time complexity analysis: measure Hamiltonian construction time for various N.
    N_values = [2, 3, 4, 5, 6]
    times = []
    dims = []
    print("\nHamiltonian construction timing:")
    for N in N_values:
        start = time.time()
        _ = construct_heisenberg_hamiltonian(N, J)
        end = time.time()
        elapsed = end - start
        times.append(elapsed)
        dims.append(2**N)
        print(f"  N = {N}, Hilbert space dim = {2**N}, time = {elapsed:.6f} sec")
    
    # Plotting the time vs. Hilbert space dimension
    plt.figure()
    plt.plot(dims, times, 'o-', label='Construction time')
    plt.xlabel('Hilbert Space Dimension (2^N)')
    plt.ylabel('Time (sec)')
    plt.title('Time Complexity of Hamiltonian Construction')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Use QR diagonalization on the test Hamiltonian
    H_diag = qr_diagonalize(H)
    # The eigenvalues are (approximately) the diagonal entries.
    eigenvalues = np.diag(H_diag)
    eigenvalues_sorted = np.sort(eigenvalues)
    print("\nEigenvalues from QR diagonalization (sorted):")
    print(eigenvalues_sorted)

# ----------------------------
# Question 3: Green's Function via LU and Cholesky decompositions
# ----------------------------
def green_function_lu(omega, H):
    """
    Computes the Green's function G from (omega*I - H) G = I
    using LU-based linear solves (np.linalg.solve internally uses LU decomposition).
    
    Parameters:
      omega : float
          The frequency parameter.
      H : ndarray
          The Hamiltonian matrix.
    
    Returns:
      G : ndarray
          The Green's function matrix.
    """
    dim = H.shape[0]
    A = omega * np.eye(dim) - H
    I = np.eye(dim)
    G = np.zeros_like(A)
    # Solve for each column of G
    for i in range(dim):
        G[:, i] = np.linalg.solve(A, I[:, i])
    return G

def green_function_cholesky(omega, H):
    """
    Computes the Green's function G from (omega*I - H) G = I
    using the Cholesky decomposition. This method requires A = (omega*I - H)
    to be positive definite.
    
    Parameters:
      omega : float
          The frequency parameter.
      H : ndarray
          The Hamiltonian matrix.
    
    Returns:
      G : ndarray or None
          The Green's function matrix if A is positive definite,
          otherwise returns None.
    """
    dim = H.shape[0]
    A = omega * np.eye(dim) - H
    try:
        # Compute lower-triangular L such that A = L L^T.
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        # A is not positive definite
        return None
    I = np.eye(dim)
    # Solve L y = I, then L^T G = y.
    y = np.linalg.solve(L, I)
    G = np.linalg.solve(L.T, y)
    return G

def construct_one_magnon_hamiltonian(N, J):
    """
    Constructs the one-magnon Hamiltonian in the subspace spanned by 
    states where exactly one spin is flipped (down) from the ferromagnetic vacuum.
    
    In this effective model:
      - Each site (where the flipped spin is located) gives a diagonal contribution J
        (coming from two bonds, each contributing J/2).
      - Hopping of the down spin to a nearest-neighbor site occurs with amplitude -J/2.
      - Periodic boundary conditions are imposed.
    
    Parameters:
      N : int
          Number of sites in the chain.
      J : float
          The coupling constant.
    
    Returns:
      H_eff : ndarray
          The effective Hamiltonian (an N x N matrix).
    """
    H_eff = np.zeros((N, N))
    for n in range(N):
        H_eff[n, n] = J
        H_eff[n, (n+1) % N] = -J/2
        H_eff[n, (n-1) % N] = -J/2
    return H_eff

def plot_green_function(N, J):
    """
    For a one-magnon Hamiltonian with N sites, computes and plots the (0,0)
    element of the Green's function (obtained via LU and Cholesky) as a function
    of frequency omega.
    
    We choose a range of omega values that spans a small interval beyond the
    eigenvalue spectrum of H.
    """
    H_magnon = construct_one_magnon_hamiltonian(N, J)
    # Compute eigenvalues of H_magnon to set an appropriate omega range.
    eigvals = np.linalg.eigvalsh(H_magnon)
    omega_min = min(eigvals) - 0.5
    omega_max = max(eigvals) + 0.5
    omegas = np.linspace(omega_min, omega_max, 400)
    
    G00_lu = []
    G00_chol = []
    for omega in omegas:
        # LU-based Green's function
        try:
            G_lu = green_function_lu(omega, H_magnon)
            G00_lu.append(G_lu[0, 0])
        except np.linalg.LinAlgError:
            G00_lu.append(np.nan)
        # Cholesky-based Green's function (if applicable)
        G_chol = green_function_cholesky(omega, H_magnon)
        if G_chol is None:
            G00_chol.append(np.nan)
        else:
            G00_chol.append(G_chol[0, 0])
    
    plt.figure()
    plt.plot(omegas, G00_lu, label="LU Green's function (G[0,0])")
    plt.plot(omegas, G00_chol, 'r--', label="Cholesky Green's function (G[0,0])")
    plt.xlabel('omega')
    plt.ylabel('G[0,0]')
    plt.title(f"Green's Function for one-magnon Hamiltonian (N = {N})")
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------------------
# Question 4: Magnon (Spin wave) energies and dispersion relation
# ----------------------------
def compute_magnon_energies(H):
    """
    Diagonalizes the one-magnon Hamiltonian H to obtain the energy eigenvalues.
    
    For a circulant one-magnon Hamiltonian, the analytical dispersion relation is:
        E(p) = 2J sin^2(p/2),
    with allowed momenta p = 2*pi*k/N, k = 0, 1, ..., N-1.
    
    This function returns the eigenvalues sorted by the corresponding momentum,
    which is estimated via the Fourier transform of each eigenvector.
    
    Parameters:
      H : ndarray
          The one-magnon Hamiltonian.
    
    Returns:
      momenta_sorted : ndarray
          The estimated momenta for the eigenstates (in radians).
      energies_sorted : ndarray
          The eigenvalues sorted by momentum.
    """
    eigvals, eigvecs = np.linalg.eigh(H)
    N = H.shape[0]
    momenta = []
    # For each eigenvector, estimate the momentum by computing its discrete Fourier transform.
    for vec in eigvecs.T:
        ft = np.fft.fft(vec)
        k = np.argmax(np.abs(ft))
        p = 2 * np.pi * k / N
        momenta.append(p)
    momenta = np.array(momenta)
    eigvals = np.array(eigvals)
    # Sort eigenvalues and momenta according to the estimated momentum.
    sort_idx = np.argsort(momenta)
    return momenta[sort_idx], eigvals[sort_idx]

def plot_magnon_dispersion(N, J):
    """
    Constructs the one-magnon Hamiltonian for N sites, computes its eigenvalues,
    and then plots the numerical magnon energies against the analytical dispersion
    relation E(p) = 2J sin^2(p/2).
    """
    H_magnon = construct_one_magnon_hamiltonian(N, J)
    momenta, energies_numeric = compute_magnon_energies(H_magnon)
    energies_analytical = 2 * J * (np.sin(momenta / 2))**2
    
    plt.figure()
    plt.plot(momenta, energies_numeric, 'bo', label='Numerical energies')
    plt.plot(momenta, energies_analytical, 'r-', label='Analytical dispersion')
    plt.xlabel('Momentum p (radians)')
    plt.ylabel('Energy E(p)')
    plt.title(f'Magnon Dispersion Relation (N = {N})')
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------------------
# Main: Testing Questions 3 and 4
# ----------------------------
if __name__ == '__main__':
    # --- Question 3: Green's Function ---
    # For demonstration we use the one-magnon Hamiltonian (N-dimensional) with N = 30.
    N_green = 30
    print(f"\n[Question 3] Plotting Green's function for one-magnon Hamiltonian with N = {N_green}")
    plot_green_function(N_green, J)
    
    # --- Question 4: Magnon Dispersion ---
    # Compute and plot the magnon energies versus momentum for N = 30.
    N_magnon = 30
    print(f"\n[Question 4] Plotting magnon dispersion for one-magnon Hamiltonian with N = {N_magnon}")
    plot_magnon_dispersion(N_magnon, J)