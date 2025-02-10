# Task 2: Heisenberg XXX Hamiltonian on a Ring

# Q1: Write a Python program that constructs the 
# Hamiltonian matrix with arbitrary number of 
# chain elements.

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.linalg import qr
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import lu_factor, lu_solve

def construct_hamiltonian(N, J=1.0):
    dim = 2 ** N
    rows = []
    cols = []
    data = []

    for s in range(dim):
        sum_sz = 0.0
        for i in range(N):
            j = (i + 1) % N
            bit_i = (s >> i) & 1
            bit_j = (s >> j) & 1
            sz_i = 0.5 if bit_i == 1 else -0.5
            sz_j = 0.5 if bit_j == 1 else -0.5
            sum_sz += sz_i * sz_j
        
        H_diag = -J * sum_sz
        rows.append(s)
        cols.append(s)
        data.append(H_diag)

        for i in range(N):
            j = (i + 1) % N
            bit_i = (s >> i) & 1
            bit_j = (s >> j) & 1

            if bit_i != bit_j:
                mask = (1 << i) | (1 << j)
                s_prime = s ^ mask
                rows.append(s)
                cols.append(s_prime)
                data.append(-J / 2)

    H = csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64)
    return H


# T(N) represents the time complexity of constructing the Hamiltonian matrix.

# Key factors:
# - The system has dim = 2^N states.
# - For each state, we iterate over N spins to compute diagonal and off-diagonal terms.
# - This results in O(N) work per state.

# Overall, this gives:
# T(N) = O(2^N * N) ≈ O(2^N), since 2^N dominates.

# Eigenvalue computation (`eigsh`):
# - The Lanczos method runs in O(m * k), where m = O(N * 2^N).
# - This adds an O(N * 2^N) term but does not change the asymptotic complexity.

# Final complexity:
# T(N) = O(2^N)

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse.linalg import eigsh

def analyze_time_complexity(num_trials=10):
    sizes = [4, 6, 8, 10, 12, 14]
    all_times = []

    for N in sizes:
        trial_times = []
        for trial in range(num_trials):
            start = time.time()
            H = construct_hamiltonian(N)
            eigsh(H, k=2, which='SA')
            trial_times.append(time.time() - start)
            print(f"N={N}, Trial {trial+1}/{num_trials}, Time taken: {trial_times[-1]:.4f} sec")

        all_times.append(trial_times)

    avg_times = [np.mean(times_for_n) for times_for_n in all_times]
    std_devs = [np.std(times_for_n) for times_for_n in all_times]

    # Compute theoretical time complexity values (normalized for visualization)
    theoretical_times = [0.0001 * (2**N) for N in sizes]  # Scale factor for better comparison

    # Plot measured times
    plt.errorbar(sizes, avg_times, yerr=std_devs, marker='o', capsize=5, label='Measured Time')

    # Plot theoretical O(2^N) curve
    plt.plot(sizes, theoretical_times, 'r--', label=r'$O(2^N)$ Theoretical')

    # Labels and title
    plt.xlabel('N (Chain Length)')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.title(f'Heisenberg XXX Model Performance Analysis ({num_trials} Trials)')
    
    plt.show()

if __name__ == "__main__":
    N = 4
    H = construct_hamiltonian(N)
    print("Hamiltonian Matrix for N=4:")
    print(H.toarray())
    analyze_time_complexity() 

H = construct_hamiltonian(2, 1.0)
print(H.toarray())

# # Q2: Write a Python program that produces the diagonal 
# # matrix for a given input Hamiltonian.

# def qr_algorithm(H, tol=1e-10, max_iter=1000):
#     H_k = H.toarray().copy()  # convert sparse matrix to dense
#     n = H_k.shape[0]
    
#     for _ in range(max_iter):
#         Q, R = qr(H_k)  # QR decomposition
#         H_k_next = R @ Q  # compute new matrix
        
#         if np.allclose(H_k, H_k_next, atol=tol):  # check for convergence
#             break
        
#         H_k = H_k_next
    
#     return np.diag(H_k)  # return the diagonal elements

# H = construct_hamiltonian(2, 1.0)
# diagonal_elements = qr_algorithm(H)
# print("Diagonal elements of the Hamiltonian:", diagonal_elements)

# # def greens_function_lu(H, omega_vals):
# #     G_vals = []
# #     I = np.eye(H.shape[0])

# #     for omega in omega_vals:
# #         A = omega * I - H
# #         lu, piv = lu_factor(A)  # LU decomposition
# #         G = lu_solve((lu, piv), I)  # Solve for G
# #         G_vals.append(np.trace(G).real)  # Take trace for visualization
    
# #     return np.array(G_vals)

# # # Generate Hamiltonian for N = 30
# # N = 30
# # H = np.random.randn(N, N)
# # H = (H + H.T) / 2  # Make it Hermitian

# # # Define frequency range
# # omega_vals = np.linspace(-5, 5, 200)

# # # Compute Green's function
# # G_lu = greens_function_lu(H, omega_vals)

# # # Plot
# # plt.plot(omega_vals, G_lu, label="LU Decomposition")
# # plt.xlabel("Frequency (ω)")
# # plt.ylabel("Tr(G)")
# # plt.title("Green's Function vs Frequency (LU)")
# # plt.legend()
# # plt.show()

# # from scipy.linalg import cho_factor, cho_solve

# # def greens_function_cholesky(H, omega_vals):
# #     G_vals = []
# #     I = np.eye(H.shape[0])

# #     for omega in omega_vals:
# #         A = omega * I - H
# #         try:
# #             c, lower = cho_factor(A)  # Cholesky decomposition
# #             G = cho_solve((c, lower), I)  # Solve for G
# #             G_vals.append(np.trace(G).real)  # Take trace for visualization
# #         except np.linalg.LinAlgError:
# #             G_vals.append(np.nan)  # Handle non-positive definite cases
    
# #     return np.array(G_vals)

# # # Compute Green's function using Cholesky
# # G_cholesky = greens_function_cholesky(H, omega_vals)

# # # Plot
# # plt.plot(omega_vals, G_cholesky, label="Cholesky Decomposition", linestyle='dashed')
# # plt.xlabel("Frequency (ω)")
# # plt.ylabel("Tr(G)")
# # plt.title("Green's Function vs Frequency (Cholesky)")
# # plt.legend()
# # plt.show()


