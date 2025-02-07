# Task 1:  Master Theorem, Time Complexity, and Strassen’s Algorithm

# Q1: Naive divide and conquer approach

# a) Use Python to construct the above native divide-and-conquer algorithm.
import numpy as np
import math
import time
import matplotlib.pyplot as plt

def naive_mm(A, B):
    n = len(A)
    
# base case: 1x1 matrix, just scalar multiplicaiton
# once recursion has reached base case, this is 
# where the actual multiplication takes place
    if n == 1:
        return np.array([[A[0][0] * B[0][0]]])
    
# check for odd dimensions and pad with zeros if necessary
    if n % 2 != 0:
        A = np.pad(A, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        B = np.pad(B, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        n += 1
    
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
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    
    return C

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

# def measure_time_complexity(max_n):
# # initializing lists to append values to
#     sizes = []
#     times = []

#     for n in range(2, max_n + 1, 2):  # increment by 2 to avoid too small values
#         # create random matrices A and B, size n x n
#         A = np.random.rand(n, n)
#         B = np.random.rand(n, n)
        
# # start timer
#         start_time = time.time()
        
# # preform matrix multiplication
#         naive_mm(A, B)
        
# # measure the elapsed time
#         elapsed_time = time.time() - start_time
# # append the n and elapsed time values
#         sizes.append(n)
#         times.append(elapsed_time)
    
#     return sizes, times

# # estimate the time complexity using the master theorem
# def master_theorem_time(n, d):
#     return n**d  

# # run the time complexity measurements for matrices of size up to 512
# max_n = 100
# sizes, times = measure_time_complexity(max_n)

# # plotting the results
# plt.figure(figsize=(10, 6))

# # plot the empirical time complexity
# plt.plot(sizes, times, label="Empirical Time Complexity (Measured)", marker='o', color='b')

# # plot the theoretical O(n^3) time complexity
# theoretical_times = [master_theorem_time(n, d) / master_theorem_time(2, d) for n in sizes]  # Normalize to start at n=2
# plt.plot(sizes, theoretical_times, label="Theoretical O(n^3)", linestyle='--', color='r')

# # set the labels and title
# plt.xlabel("Matrix Size (n)")
# plt.ylabel("Time (seconds)")
# plt.title("Comparison of Empirical Time Complexity and Theoretical O(n^3)")

# plt.legend()
# plt.grid(True)
# plt.show()


def measure_time_complexity(max_n, num_trials=3):
    sizes, times = [], []
    
    for n in range(2, max_n + 1, 2):
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        # Run multiple trials and average
        trial_times = []
        for _ in range(num_trials):
            start_time = time.perf_counter()
            naive_mm(A, B)
            elapsed_time = time.perf_counter() - start_time
            trial_times.append(elapsed_time)

        avg_time = sum(trial_times) / num_trials
        sizes.append(n)
        times.append(avg_time)
    
    return sizes, times

def master_theorem_time(n, d):
    return n**d  

# Run time complexity measurement
max_n = 100
sizes, times = measure_time_complexity(max_n)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(sizes, times, label="Empirical Time Complexity (Measured)", marker='o', color='b')

# Plot theoretical O(n^3) complexity
theoretical_times = [master_theorem_time(n, d) / master_theorem_time(2, d) for n in sizes]
plt.plot(sizes, theoretical_times, label="Theoretical O(n^3)", linestyle='--', color='r')

plt.xlabel("Matrix Size (n)")
plt.ylabel("Time (seconds)")
plt.title("Comparison of Empirical Time Complexity and Theoretical O(n^3)")
plt.legend()
plt.grid(True)
plt.show()

# Q2) Strassen’s algorithm

# M_1 = (A_11+A_22)(B_11+B_22)
# M2 = (A_21 + A_22)B_11
# M3 = A_11(B_12 − B_22) 
# M4 = A_22(B_21 − B_11)
# M5 = (A_11 + A_12)B_22
# M6 = (A_21 − A_11)(B_11 + B_12)
# M7 = (A_12 − A_22)(B_21 + B_22)

# C_11 = M_1 + M_4 − M_5 + M_7, 
# C_12 = M_3 + M_5
# C_21 = M_2 + M_4
# C_22 = M_1 − M_2 + M_3 + M_6