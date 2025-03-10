#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Common: Define the target probability density function (PDF)
# =============================================================================
def target_pdf(t, a, b):
    """
    Target probability density function for t ≥ 0:
        p(t) = exp(-b*t) * cos²(a*t)
    
    Parameters:
        t : float or np.array
            Time variable.
        a : float
            Parameter a (e.g., 4).
        b : float
            Parameter b (e.g., 4).
    
    Returns:
        p(t) evaluated at t.
    """
    return np.exp(-b * t) * (np.cos(a * t))**2

# =============================================================================
# a) Rejection Sampling using Uniform Proposal
# =============================================================================
#
# We wish to sample from p(t) = exp(-4t)*cos²(4t) for t ≥ 0.
# We use a uniform proposal q(t) ~ U(0, t_f). To ensure most of the probability
# mass is captured, we select t_f such that p(t) is negligible for t > t_f.
# For example, with b = 4 the exponential decay is rapid; here we choose t_f = 2.
# For a uniform proposal on [0, t_f], q(t) = 1/t_f. Since p(0) = 1, the maximum
# of p(t) in [0, t_f] is 1. Hence we can take the envelope constant M = t_f,
# so that M * q(t) = 1, and the acceptance condition becomes:
#
#   Accept candidate t if u < p(t),
#
# where u ~ U(0,1).

def rejection_sampling_uniform(N, t_f, a, b):
    """
    Perform rejection sampling using a uniform proposal on [0, t_f].
    
    Parameters:
        N   : int
              Number of accepted samples desired.
        t_f : float
              Upper bound of the uniform proposal.
        a,b : floats
              Parameters in the target PDF.
              
    Returns:
        accepted_samples : np.array
                           Array of N accepted samples.
        rejection_ratio  : float
                           Ratio of accepted samples to rejected samples.
    """
    accepted = []
    total_trials = 0
    while len(accepted) < N:
        # Sample candidate t from Uniform(0, t_f)
        t = np.random.uniform(0, t_f)
        total_trials += 1
        # Draw u uniformly from [0,1]
        u = np.random.uniform(0, 1)
        # Acceptance condition: u < p(t) (since M*q(t) = 1)
        if u < target_pdf(t, a, b):
            accepted.append(t)
    accepted = np.array(accepted)
    rejection_ratio = len(accepted) / (total_trials - len(accepted))
    return accepted, rejection_ratio

# =============================================================================
# b) Rejection Sampling using Exponential Proposal
# =============================================================================
#
# Now we use an exponential proposal distribution. Here we choose
# q(t) ~ Exp(λ) with λ = 2, which has PDF:
#
#   q(t) = 2 exp(-2t),   t ≥ 0.
#
# We again wish to sample from p(t) = exp(-4t)*cos²(4t) with a = 4, b = 4.
#
# To apply rejection sampling we must choose an envelope constant M such that
#   p(t) ≤ M * q(t)   for all t ≥ 0.
#
# At t = 0: p(0)=1 and q(0)=2, so we require M ≥ 1/2. For t > 0 the ratio
#   p(t) / q(t) = [exp(-4t)*cos²(4t)] / [2 exp(-2t)] = (exp(-2t)*cos²(4t))/2,
# which is ≤ 1/2. Thus, we can choose M = 1/2.
#
# With M = 1/2, the acceptance condition becomes:
#
#   Accept candidate t if u < p(t) / (M*q(t)) = p(t) / (exp(-2t))
#         = exp(-4t)*cos²(4t) / exp(-2t)
#         = exp(-2t)*cos²(4t).
#
# We generate candidates t from the exponential distribution (using numpy’s
# exponential sampler with scale = 1/λ = 0.5).

def rejection_sampling_exponential(N, a, b):
    """
    Perform rejection sampling using an exponential proposal distribution.
    
    The proposal is q(t) = 2 exp(-2t) (i.e., Exp(λ=2)), and we use an envelope
    constant M = 1/2.
    
    Parameters:
        N  : int
             Number of accepted samples desired.
        a,b: floats
             Parameters in the target PDF.
    
    Returns:
        accepted_samples : np.array
                           Array of N accepted samples.
        rejection_ratio  : float
                           Ratio of accepted samples to rejected samples.
    """
    accepted = []
    total_trials = 0
    M = 0.5  # Envelope constant
    while len(accepted) < N:
        # Sample candidate t from the exponential distribution with λ = 2
        t = np.random.exponential(scale=0.5)  # scale = 1/λ
        total_trials += 1
        u = np.random.uniform(0, 1)
        # q(t) = 2 exp(-2t) and M*q(t) = exp(-2t)
        # Acceptance condition: u < p(t) / (M*q(t)) = exp(-4t)*cos²(4t) / exp(-2t)
        if u < target_pdf(t, a, b) / np.exp(-2 * t):
            accepted.append(t)
    accepted = np.array(accepted)
    rejection_ratio = len(accepted) / (total_trials - len(accepted))
    return accepted, rejection_ratio

# =============================================================================
# Main: Run and Plot Histograms for Various Sample Sizes for Both Methods
# =============================================================================
if __name__ == '__main__':
    # Parameters for the target PDF: a = 4, b = 4
    a = 4
    b = 4
    # For the uniform proposal, choose t_f such that most of the mass is captured.
    # With b=4, the exponential decays rapidly, so we use t_f = 2.
    t_f = 2.0
    
    # Define sample sizes to test.
    sample_sizes = [100, 1000, 10000]
    
    # ----------------------------
    # Part a: Uniform Proposal Sampling
    # ----------------------------
    plt.figure(figsize=(12, 4))
    for i, N in enumerate(sample_sizes):
        samples, ratio = rejection_sampling_uniform(N, t_f, a, b)
        print(f"Uniform Proposal: N = {N}, Rejection Ratio (accepts/rejects) = {ratio:.4f}")
        plt.subplot(1, len(sample_sizes), i+1)
        plt.hist(samples, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='k')
        plt.title(f"Uniform Proposal, N = {N}")
        plt.xlabel("t")
        plt.ylabel("Density")
    plt.tight_layout()
    plt.show()
    
    # ----------------------------
    # Part b: Exponential Proposal Sampling
    # ----------------------------
    plt.figure(figsize=(12, 4))
    for i, N in enumerate(sample_sizes):
        samples, ratio = rejection_sampling_exponential(N, a, b)
        print(f"Exponential Proposal: N = {N}, Rejection Ratio (accepts/rejects) = {ratio:.4f}")
        plt.subplot(1, len(sample_sizes), i+1)
        plt.hist(samples, bins=30, density=True, alpha=0.7, color='lightgreen', edgecolor='k')
        plt.title(f"Exponential Proposal, N = {N}")
        plt.xlabel("t")
        plt.ylabel("Density")
    plt.tight_layout()
    plt.show()