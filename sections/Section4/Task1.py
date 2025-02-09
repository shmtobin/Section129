# Task 1:  Master Theorem, Time Complexity, and Strassen’s Algorithm

# Q1: Naive divide and conquer approach

# a) Use Python to construct the above native divide-and-conquer algorithm.
import numpy as np
import math
import time
import matplotlib.pyplot as plt

def naive_mm(A, B):
    n = A.shape[0]
    
# base case: 1x1 matrix, just scalar multiplicaiton
# once recursion has reached base case, this is 
# where the actual multiplication takes place
    if n == 1:
        return np.array([[A[0][0] * B[0][0]]])
    
# splitting matrices into 4 n/2 * n/2 submatrices
    mid = n // 2
# selecting top left, top right, bottom left, bottom right submatrices of A
    A11, A12, A21, A22 = A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]
# same for B
    B11, B12, B21, B22 = B[:mid, :mid], B[:mid, mid:], B[mid:, :mid], B[mid:, mid:]
    
# computes the submatrices of C, calling the function
# within itself (recursion) to iteratively perform this
# until the base case is reached

    C11 = naive_mm(A11, B11) + naive_mm(A12, B21)
    C12 = naive_mm(A11, B12) + naive_mm(A12, B22)
    C21 = naive_mm(A21, B11) + naive_mm(A22, B21)
    C22 = naive_mm(A21, B12) + naive_mm(A22, B22)
    
# combines the results into a single matrix
    top = np.hstack((C11, C12))
    bottom = np.hstack((C21, C22))
    return np.vstack((top, bottom))

# example usage
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = naive_mm(A, B)
print(C)

# b) Use the master theorem to determine the time complexity, and find the
# critical exponent.

# need to determine the recurrence relation

# T(n) = aT(n/b) + f(n)
# a = number of recursive cells = 8
# b = factor by which the size of each subproblem is reduced = 2
# f(n) = cost of dividing the problem and combining the results = O(n^d)
# d = 2

# for f(n),  we can consider how efficient our matrix division 
# is, which is O(n^2) because our indexing/slicing is O(1) per 
# element, and there are n^2 elements in our square matrix
# so d = 2

# T(n) = aT(n/b) + O(n^d)
# there are 3 cases to consider with this
# 1) d>log_b(a) => T(n) = O(n^d)
# 2) d=log_b(a) => T(n) = O(n^d * log(n))
# 3) d<log_b(a) => T(n) = O(n^log_b(a))

# our values for a, b, and d leave us in case 3

# T(n) = O(n^log_b(a))=O(n^log_2(8))=O(n^3)
# so 3 is the critical exponent

def master_thm(a, b, d):
    
    log_b_a = math.log(a, b)  
    
    print(f"Master Theorem Parameters: a = {a}, b = {b}, d = {d}")
    print(f"log_b(a) = log_{b}({a}) = {log_b_a:.2f}")
    
    if d > log_b_a:
        print(f"Since d ({d}) > log_b(a) ({log_b_a:.2f}), T(n) = O(n^{d}) = O(n^{d})")
    elif d == log_b_a:
        print(f"Since d ({d}) = log_b(a) ({log_b_a:.2f}), T(n) = O(n^{d} log n) = O(n^{d} log n)")
    else:
        print(f"Since d ({d}) < log_b(a) ({log_b_a:.2f}), T(n) = O(n^{log_b_a}) = O(n^{log_b_a:.2f})")
    
    print(f"The critical exponent is {log_b_a:.2f} and the time complexity is O(n^{log_b_a:.2f}) = O(n^3).")

a = 8  
b = 2  
d = 2  
master_thm(a, b, d)

# c) Compare and visualize the time complexity obtained from your Python
# program and the time complexity calculated from the master theorem. Do
# the asymptotic behaviors agree?

def measure_time_complexity_naive(max_n):
    sizes = []
    times = []
    n_values = [2**p for p in range(1, int(np.log2(max_n)) + 1)]
    for n in n_values:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        num_trials = 5
        total_time = 0
        for _ in range(num_trials):
            start = time.perf_counter()
            naive_mm(A, B)
            total_time += time.perf_counter() - start
        avg_time = total_time / num_trials
        sizes.append(n)
        times.append(avg_time)
        print(f"Naive n={n}, time={avg_time:.6f}s")
    return sizes, times

max_n = 256
def master_theorem_time(n, d):
    return n**d  

sizes, times = measure_time_complexity_naive(max_n)


# empirical times
t0 = times[0]
normalized_empirical = [t / t0 for t in times]

n_dense = np.linspace(2, max_n, 100)  # 100 points for smoothness
theoretical_dense = [(n/2)**3 for n in n_dense]  # Normalized to n=2

# plotting
plt.figure(figsize=(10, 6))

# empirical data (normalized)
t0 = times[0]
normalized_empirical = [t/t0 for t in times]

plt.plot(sizes, normalized_empirical, 'bo-', label="Empirical (Normalized)")
plt.plot(n_dense, theoretical_dense, 'r--', label="Theoretical O(n³)")

plt.xlabel("Matrix Size (n)")
plt.ylabel("Normalized Time")
plt.title("Time Complexity Analysis (5 Trials per Measurement)")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(f'Plots/naive_mm_time_complexity.png')

# Q2) Strassen’s algorithm

# a) Use Python to construct the above Strassen’s algorithm.

# M_1 = (A_11+A_22)(B_11+B_22)
# M_2 = (A_21 + A_22)B_11
# M_3 = A_11(B_12 − B_22) 
# M_4 = A_22(B_21 − B_11)
# M_5 = (A_11 + A_12)B_22
# M_6 = (A_21 − A_11)(B_11 + B_12)
# M_7 = (A_12 − A_22)(B_21 + B_22)

# C_11 = M_1 + M_4 − M_5 + M_7, 
# C_12 = M_3 + M_5
# C_21 = M_2 + M_4
# C_22 = M_1 − M_2 + M_3 + M_6

def strassen_mm(A, B):
    n = A.shape[0]
    
# base case: 1x1 matrix
    if n == 1:
        return np.array([[A[0, 0] * B[0, 0]]])
    
# split matrices into submatrices, handling odd dimensions
    mid = n // 2
    
# define submatrix splits (handles odd n)
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:] if mid*2 == n else A[:mid, mid:n]
    A21 = A[mid:, :mid] if mid*2 == n else A[mid:n, :mid]
    A22 = A[mid:, mid:] if mid*2 == n else A[mid:n, mid:n]
    
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:] if mid*2 == n else B[:mid, mid:n]
    B21 = B[mid:, :mid] if mid*2 == n else B[mid:n, :mid]
    B22 = B[mid:, mid:] if mid*2 == n else B[mid:n, mid:n]
    
# compute Strassen's intermediate matrices
    M1 = strassen_mm(A11 + A22, B11 + B22)
    M2 = strassen_mm(A21 + A22, B11)
    M3 = strassen_mm(A11, B12 - B22)
    M4 = strassen_mm(A22, B21 - B11)
    M5 = strassen_mm(A11 + A12, B22)
    M6 = strassen_mm(A21 - A11, B11 + B12)
    M7 = strassen_mm(A12 - A22, B21 + B22)
    
# compute C submatrices
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
# combine results, handling odd dimensions
    top = np.hstack((C11, C12)) if C12.shape[1] == C11.shape[1] else np.hstack((C11, np.zeros((C11.shape[0],1))))
    bottom = np.hstack((C21, C22)) if C22.shape[1] == C21.shape[1] else np.hstack((C21, np.zeros((C21.shape[0],1))))
    C = np.vstack((top, bottom)) if bottom.shape[0] == top.shape[0] else np.vstack((top, np.zeros((1, top.shape[1]))))
    
    return C[:n, :n]  

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = strassen_mm(A, B)
print(C) 

# b) Find a, b, f of the recursive expression.

# T(n) = aT(n/b) + f(n)
# a = number of recursive multiplications = 7
# b = factor by which the size of each subproblem is reduced = 2
# f(n) = cost of matrix additions/subtractions and combining results = O(n^d)
# d = 2

# For f(n), we consider the cost of matrix operations:
# - Matrix additions (e.g., A11 + A22) and subtractions (e.g., B12 − B22)
# - Combining results into C11, C12, C21, C22
# Each operation is O(n^2) because they involve element-wise arithmetic on n x n matrices.

# T(n) = 7T(n/2) + O(n^2)

# c) With the above recursion expression, use the master theorem to determine
# the time complexity, and find the critical exponent.
# Master theorem cases:
# 1) d > log_b(a) → T(n) = O(n^d)
# 2) d = log_b(a) → T(n) = O(n^d log n)
# 3) d < log_b(a) → T(n) = O(n^{log_b(a)})

# For Strassen’s algorithm:
# log_b(a) = log_2(7) ≈ 2.81
# Since d = 2 < 2.81, we are in Case 3.

# T(n) = O(n^{log_b(a)}) = O(n^{log_2(7)}) ≈ O(n^{2.81})
# The critical exponent is log_2(7) ≈ 2.81.

def measure_time_complexity_strassen(max_n):
    sizes = []
    times = []
    n_values = [2**p for p in range(1, int(np.log2(max_n)) + 1)]
    for n in n_values:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        num_trials = 5
        total_time = 0
        for _ in range(num_trials):
            start = time.perf_counter()
            strassen_mm(A, B)
            total_time += time.perf_counter() - start
        avg_time = total_time / num_trials
        sizes.append(n)
        times.append(avg_time)
        print(f"Strassen n={n}, time={avg_time:.6f}s")
    return sizes, times

def theoretical_time(n, exponent, base_n=2):
    return (n / base_n) ** exponent

max_n = 256
sizes_naive, times_naive = measure_time_complexity_naive(max_n)
sizes_strassen, times_strassen = measure_time_complexity_strassen(max_n)

# normalize data
t0_naive = times_naive[0]
normalized_naive = [t/t0_naive for t in times_naive]

t0_strassen = times_strassen[0]
normalized_strassen = [t/t0_strassen for t in times_strassen]

# generate smooth theoretical curves
n_dense = np.linspace(2, max_n, 100)
theoretical_naive = [theoretical_time(n, 3) for n in n_dense]  # O(n³)
theoretical_strassen = [theoretical_time(n, math.log2(7)) for n in n_dense]  # O(n^2.81)

# plotting
plt.figure(figsize=(12, 7))

# empirical data
plt.plot(sizes_naive, normalized_naive, 'bo-', label="Naive (Empirical)")
plt.plot(sizes_strassen, normalized_strassen, 'go-', label="Strassen (Empirical)")

# theoretical curves
plt.plot(n_dense, theoretical_naive, 'b--', label="Theoretical O(n³)")
plt.plot(n_dense, theoretical_strassen, 'g--', label="Theoretical O(n²·⁸¹)")

plt.xlabel("Matrix Size (n)")
plt.ylabel("Normalized Time")
plt.title("Time Complexity: Naive vs. Strassen's Algorithm")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(f'Plots/naive_v_strassen_comparison.png')


# the results of this plot demonstrate the increased efficiency
# of the Strassen algorithm over the naive approach